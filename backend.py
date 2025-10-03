import os
import io
import logging
import time
import pickle
import faiss
from deepface import DeepFace
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, UploadFile, HTTPException, Depends, Response
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request as GoogleAuthRequest
import secrets
import json
from typing import Dict

load_dotenv() 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# In-memory storage for testing (use database in production)
sessions: Dict[str, Dict] = {}  # session_id: {'state': str, 'credentials': str}
events: Dict[str, Dict] = {}  # event_id: {'folder_id': str, 'index_file': str, 'metadata_file': str, 'session_id': str}

def get_drive_service(credentials_dict):
    try:
        logging.info("Attempting to create Drive service")
        credentials = Credentials(**json.loads(credentials_dict))
        if credentials.expired and credentials.refresh_token:
            logging.info("Refreshing expired credentials")
            credentials.refresh(GoogleAuthRequest())
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        logging.error(f"Error creating Drive service: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create Drive service: {str(e)}")

def build_faiss_index(folder_id: str, credentials_dict: str):
    start_time = time.time()
    logging.info(f"Starting build_faiss_index for folder_id: {folder_id}")
    
    try:
        drive_service = get_drive_service(credentials_dict)
        logging.info("Drive service initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Drive service: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize Drive service: {str(e)}")

    query = f"'{folder_id}' in parents and (mimeType='image/jpeg' or mimeType='image/png')"
    try:
        logging.info(f"Listing files in folder_id: {folder_id}")
        results = drive_service.files().list(q=query, fields="files(id, name, mimeType)").execute()
        files = results.get('files', [])
        if not files:
            logging.warning(f"No images found in folder_id: {folder_id}")
            raise HTTPException(status_code=400, detail="No JPEG/PNG images found in the selected folder.")
        logging.info(f"Files found in Drive folder: {[(f['name'], f['mimeType']) for f in files]}")
    except Exception as e:
        logging.error(f"Error listing Drive files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list Drive files: {str(e)}")

    all_embeddings = []
    embedding_to_image = []

    for file in files:
        try:
            logging.info(f"Downloading file: {file['name']} (ID: {file['id']})")
            request = drive_service.files().get_media(fileId=file['id'])
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.seek(0)

            logging.info(f"Processing image: {file['name']}")
            event_image = np.array(Image.open(fh))
            event_objs = DeepFace.represent(event_image, model_name='ArcFace', detector_backend='retinaface', enforce_detection=False)
            if not event_objs:
                logging.info(f"No faces detected in {file['name']}, skipping.")
                continue

            for obj in event_objs:
                all_embeddings.append(obj['embedding'])
                embedding_to_image.append((file['id'], file['name']))

            logging.info(f"Processed {file['name']}: {len(event_objs)} embeddings added.")
        except Exception as e:
            logging.error(f"Error processing {file['name']}: {str(e)}")
            continue

    if not all_embeddings:
        logging.warning("No valid face embeddings found in any images.")
        raise HTTPException(status_code=400, detail="No valid face embeddings found in Drive images.")

    try:
        logging.info("Creating FAISS index")
        embeddings_matrix = np.array(all_embeddings).astype('float32')
        dimension = embeddings_matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings_matrix)
        index.add(embeddings_matrix)
        logging.info(f"FAISS index created with {len(all_embeddings)} embeddings")
    except Exception as e:
        logging.error(f"Error building FAISS index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to build FAISS index: {str(e)}")

    event_id = secrets.token_hex(8)
    index_file = f'face_index_{event_id}.faiss'
    metadata_file = f'face_metadata_{event_id}.pkl'
    try:
        logging.info(f"Saving FAISS index to {index_file} and metadata to {metadata_file}")
        faiss.write_index(index, index_file)
        with open(metadata_file, 'wb') as f:
            pickle.dump(embedding_to_image, f)
        logging.info(f"Saved FAISS index and metadata successfully")
    except Exception as e:
        logging.error(f"Error saving FAISS index or metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save FAISS index or metadata: {str(e)}")

    logging.info(f"FAISS index built with {index.ntotal} embeddings. Time taken: {time.time() - start_time:.2f} seconds.")
    return event_id, index_file, metadata_file

@app.get("/", response_class=HTMLResponse)
async def admin_panel():
    try:
        logging.info("Serving admin panel")
        with open("admin.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        logging.error("admin.html not found")
        raise HTTPException(status_code=500, detail="Admin page not found")

@app.get("/client/{event_id}", response_class=HTMLResponse)
async def client_page(event_id: str):
    if event_id not in events:
        logging.error(f"Invalid event_id: {event_id}")
        raise HTTPException(status_code=404, detail="Invalid Event")
    try:
        logging.info(f"Serving client page for event_id: {event_id}")
        with open("client.html", "r") as f:
            html = f.read().replace("{{event_id}}", event_id)
            return html
    except FileNotFoundError:
        logging.error("client.html not found")
        raise HTTPException(status_code=500, detail="Client page not found")

@app.get("/auth")
async def auth(request: Request):
    try:
        client_secrets_file = os.getenv('CLIENT_SECRETS_FILE')
        if not client_secrets_file:
            logging.error("CLIENT_SECRETS_FILE not set in .env")
            raise HTTPException(status_code=500, detail="Client secrets file not configured in .env")
        if not os.path.exists(client_secrets_file):
            logging.error(f"Client secrets file not found at: {client_secrets_file}")
            raise HTTPException(status_code=500, detail=f"Client secrets file not found at: {client_secrets_file}")
        
        logging.info("Creating OAuth flow")
        flow = Flow.from_client_secrets_file(
            client_secrets_file,
            scopes=SCOPES,
            redirect_uri=f"{request.url.scheme}://{request.url.netloc}/oauth_callback"
        )
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        session_id = secrets.token_hex(16)
        sessions[session_id] = {'state': state}
        response = Response(content=json.dumps({"auth_url": authorization_url}))
        response.set_cookie(key="session_id", value=session_id, httponly=True, samesite="lax")
        logging.info(f"Auth URL generated for session_id: {session_id}")
        return response
    except Exception as e:
        logging.error(f"Error in /auth endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Authentication setup failed: {str(e)}")

@app.get("/oauth_callback")
async def oauth_callback(request: Request, code: str = None, state: str = None, error: str = None):
    if error or not code:
        logging.error(f"OAuth error: {error or 'No code provided'}")
        raise HTTPException(status_code=400, detail=f"Authentication failed: {error or 'User canceled authentication'}")
    try:
        session_id = request.cookies.get("session_id")
        if not session_id or session_id not in sessions or sessions[session_id].get('state') != state:
            logging.error(f"Invalid session or state in oauth_callback: session_id={session_id}, state={state}")
            raise HTTPException(status_code=401, detail="Invalid state or session")
        
        client_secrets_file = os.getenv('CLIENT_SECRETS_FILE')
        if not client_secrets_file or not os.path.exists(client_secrets_file):
            logging.error(f"Client secrets file not found at: {client_secrets_file}")
            raise HTTPException(status_code=500, detail="Client secrets file not found or not configured.")
        
        logging.info("Fetching OAuth token")
        flow = Flow.from_client_secrets_file(
            client_secrets_file,
            scopes=SCOPES,
            state=state,
            redirect_uri=f"{request.url.scheme}://{request.url.netloc}/oauth_callback"
        )
        flow.fetch_token(code=code)
        credentials = flow.credentials
        credentials_dict = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        }
        sessions[session_id]['credentials'] = json.dumps(credentials_dict)
        logging.info(f"Credentials stored for session_id: {session_id}, refresh_token: {credentials_dict.get('refresh_token')}")
        response = RedirectResponse(url="/")
        response.set_cookie(key="session_id", value=session_id, httponly=True, samesite="lax")
        return response
    except Exception as e:
        logging.error(f"Error in /oauth_callback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OAuth callback failed: {str(e)}")

def get_session(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        logging.error(f"Unauthorized access: Invalid or missing session_id: {session_id}")
        raise HTTPException(status_code=401, detail="Unauthorized")
    return sessions[session_id], session_id

@app.get("/list_folders")
async def list_folders(session_data: dict = Depends(get_session)):
    session, session_id = session_data
    if 'credentials' not in session:
        logging.error(f"No credentials in session for list_folders, session_id: {session_id}")
        raise HTTPException(status_code=401, detail="Please connect to Google Drive first.")
    credentials_dict = session['credentials']
    try:
        logging.info(f"Listing folders for session_id: {session_id}")
        drive_service = get_drive_service(credentials_dict)
        results = drive_service.files().list(
            q="mimeType='application/vnd.google-apps.folder'",
            fields="files(id, name)"
        ).execute()
        folders = results.get('files', [])
        logging.info(f"Found {len(folders)} folders in Drive for session_id: {session_id}")
        return {"folders": [{"id": f['id'], "name": f['name']} for f in folders]}
    except Exception as e:
        logging.error(f"Error listing Drive folders: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list Drive folders: {str(e)}")

@app.post("/create_event")
async def create_event(folder_id: str = Form(...), session_data: dict = Depends(get_session)):
    session, session_id = session_data
    if 'credentials' not in session:
        logging.error(f"No credentials in session for create_event, session_id: {session_id}")
        raise HTTPException(status_code=401, detail="Please connect to Google Drive first.")
    credentials_dict = session['credentials']
    logging.info(f"Received create_event request for folder_id: {folder_id}, session_id: {session_id}")
    try:
        event_id, index_file, metadata_file = build_faiss_index(folder_id, credentials_dict)
        events[event_id] = {
            'folder_id': folder_id,
            'index_file': index_file,
            'metadata_file': metadata_file,
            'session_id': session_id
        }
        logging.info(f"Event created: event_id={event_id}, folder_id={folder_id}")
        return {"event_id": event_id, "message": "Index built successfully."}
    except ValueError as ve:
        logging.error(f"ValueError in create_event: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Error creating event: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to build index: {str(e)}")

@app.get("/list_events")
async def list_events(session_data: dict = Depends(get_session)):
    session, session_id = session_data
    try:
        user_events = [
            {"event_id": eid, "folder_id": edata['folder_id']}
            for eid, edata in events.items() if edata['session_id'] == session_id
        ]
        logging.info(f"Listed {len(user_events)} events for session_id: {session_id}")
        return {"events": user_events}
    except Exception as e:
        logging.error(f"Error listing events: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list events: {str(e)}")

@app.post("/delete_event")
async def delete_event(event_id: str = Form(...), session_data: dict = Depends(get_session)):
    session, session_id = session_data
    if event_id not in events:
        raise HTTPException(status_code=404, detail="Event not found")
    event_data = events[event_id]
    if event_data['session_id'] != session_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this event")
    try:
        os.remove(event_data['index_file'])
        os.remove(event_data['metadata_file'])
        del events[event_id]
        logging.info(f"Event deleted: event_id={event_id}")
        return {"message": "Event deleted successfully"}
    except Exception as e:
        logging.error(f"Error deleting event {event_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete event: {str(e)}")

@app.get("/validate_event")
async def validate_event(event_id: str, session_data: dict = Depends(get_session)):
    session, session_id = session_data
    if event_id in events and events[event_id]['session_id'] == session_id:
        return {"valid": True}
    raise HTTPException(status_code=404, detail="Event not found or not authorized")

@app.post("/process_selfie")
async def process_selfie(event_id: str = Form(...), selfie: UploadFile = Form(...)):
    if event_id not in events:
        logging.error(f"Invalid event_id: {event_id}")
        raise HTTPException(status_code=404, detail="Invalid event")
    if not selfie:
        logging.error("No selfie uploaded")
        raise HTTPException(status_code=400, detail="No selfie uploaded.")
    user_photo_bytes = await selfie.read()

    event_data = events[event_id]
    session_id = event_data['session_id']
    if session_id not in sessions:
        logging.error(f"Session expired for session_id: {session_id}")
        raise HTTPException(status_code=500, detail="Session expired")
    credentials_dict = sessions[session_id]['credentials']
    drive_service = get_drive_service(credentials_dict)

    try:
        logging.info("Processing selfie")
        user_image = np.array(Image.open(io.BytesIO(user_photo_bytes)))
        user_objs = DeepFace.represent(user_image, model_name='ArcFace', detector_backend='retinaface', enforce_detection=False)
        if not user_objs:
            logging.info("No face detected in selfie")
            raise HTTPException(status_code=400, detail="No face detected in selfie.")
        user_embedding = np.array([user_objs[0]['embedding']]).astype('float32')
        faiss.normalize_L2(user_embedding)
    except Exception as e:
        logging.error(f"Error processing selfie: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing selfie: {str(e)}")

    index = faiss.read_index(event_data['index_file'])
    with open(event_data['metadata_file'], 'rb') as f:
        embedding_to_image = pickle.load(f)

    k = index.ntotal
    distances, indices = index.search(user_embedding, k)

    matched_photos = {}
    COSINE_THRESHOLD = 0.68

    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        if dist >= (1-COSINE_THRESHOLD): #COSINE_THRESHOLD
            file_id, file_name = embedding_to_image[idx]
            if file_name not in matched_photos or matched_photos[file_name][1] < dist:
                matched_photos[file_name] = (file_id, float(dist))  # Convert numpy.float32 to float

    matched_photos_list = [
        {'file_id': fid, 'file_name': fname, 'distance': float(dist)}  # Convert distance to float
        for fname, (fid, dist) in matched_photos.items()
    ]
    matched_photos_list = sorted(matched_photos_list, key=lambda x: x['distance'], reverse=True)

    logging.info(f"Found {len(matched_photos_list)} matching photos for event_id: {event_id}")
    return {"matches": matched_photos_list}

@app.get("/image/{file_id}")
async def serve_image(file_id: str):
    for event_data in events.values():
        session_id = event_data['session_id']
        if session_id not in sessions:
            continue
        credentials_dict = sessions[session_id]['credentials']
        drive_service = get_drive_service(credentials_dict)
        try:
            logging.info(f"Serving image with file_id: {file_id}")
            request = drive_service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.seek(0)
            
            image = Image.open(fh)
            if image.size[0] > 1000 or image.size[1] > 1000:
                image = image.resize((1000, int(1000 * image.size[1] / image.size[0])), Image.LANCZOS)
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=85)
            output.seek(0)
            return StreamingResponse(output, media_type="image/jpeg")
        except Exception as e:
            logging.error(f"Error serving image {file_id}: {str(e)}")
            continue
    logging.error(f"Image not found: {file_id}")
    raise HTTPException(status_code=404, detail="Image not found")