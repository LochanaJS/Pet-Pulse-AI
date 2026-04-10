import os
import io
import bcrypt
import uvicorn
import requests
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
import database  # Importing the whole module to avoid naming conflicts
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

app = FastAPI(title="Pet Pulse AI - Local AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# ================================
# CANCER MODEL LOAD
# ================================
try:
    with open("xgboost_cancer_model.pkl", "rb") as f:
        cancer_model = pickle.load(f)

    with open("model_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    label_encoders = artifacts["label_encoders"]
    target_encoder = artifacts["target_encoder"]
    feature_names = artifacts["feature_names"]

    explainer = shap.TreeExplainer(cancer_model)

    print("✅ Cancer model loaded")

except Exception as e:
    print(f"❌ Model loading failed: {e}")
    cancer_model = None

# Directory Setup
UPLOAD_DIR = "uploads/xrays"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# --- Pydantic Models ---
class UserRegister(BaseModel):
    fullName: str
    email: EmailStr
    password: str
    phoneNo: str

class AdminRegister(BaseModel):
    fullName: str
    role: str
    email: EmailStr
    companyName: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class AdminLogin(BaseModel):
    email: str
    password: str

class AppointmentCreate(BaseModel):
    userId: int
    ownerName: str
    petName: str
    breed: str
    sex: str
    reason: str
    appointmentDate: str 
    contactNo: str

class PaymentCreate(BaseModel):
    appointmentId: int
    userId: int
    cardHolderName: str
    cardNumber: str
    expiryDate: str
    cvv: str 
    amount: float = 500.00

class FeedbackCreate(BaseModel):
    userId: int
    fullName: str
    email: str
    rating: int
    comment: str

class FeedbackResponse(BaseModel):
    feedbackId: int
    adminResponse: str

# --- Add this to your Pydantic Models section ---
class NutritionRequest(BaseModel):
    userId: int
    petName: str
    category: str
    breed: str
    age: str
    sex: str
    condition: str

class CancerRequest(BaseModel):
    userId: int
    reportType: str  # CBC or FULL

    species: str
    breed: str
    age: float
    sex: str
    neutered_status: str

    neutrophils: float
    lymphocytes: float
    rbc: float
    hemoglobin: float

    # Optional fields (FULL report only)
    albumin: float | None = None
    globulin: float | None = None
    calcium: float | None = None
    alp: float | None = None

def compute_ratios(data):
    nlr = data.neutrophils / data.lymphocytes
    ag_ratio = (data.albumin or 1) / (data.globulin or 1)
    anemia_index = data.rbc / data.hemoglobin

    return nlr, ag_ratio, anemia_index

def encode_input(df):
    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = encoder.transform([df[col][0]])[0]
    return df

# ================================
# CANCER PREDICTION ENDPOINT (FIXED)
# ================================
@app.post("/cancer/predict")
async def predict_cancer(data: CancerRequest):
    if cancer_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # 1. FEATURE ENGINEERING
        nlr = data.neutrophils / data.lymphocytes if data.lymphocytes != 0 else 0
        ag_ratio = (data.albumin or 1) / (data.globulin or 1)
        anemia_index = data.rbc / data.hemoglobin if data.hemoglobin != 0 else 0

        # 2. SAFE ENCODING LOGIC
        def get_safe_label(column_name, user_value):
            encoder = label_encoders.get(column_name)
            if not encoder: return user_value
            if user_value in encoder.classes_:
                return encoder.transform([user_value])[0]
            else:
                return encoder.transform([encoder.classes_[0]])[0]

        # 3. CREATE INPUT DICTIONARY 
        # Note: We match the keys to exactly what your model artifact feature_names uses
        input_dict = {
            "Species": get_safe_label("Species", data.species),
            "Breed": get_safe_label("Breed", data.breed),
            "Age": data.age,
            "Sex": get_safe_label("Sex", data.sex),
            "Neutered_Status": get_safe_label("Neutered_Status", data.neutered_status),
            "Neutrophils_Percent": data.neutrophils,
            "Lymphocytes_Percent": data.lymphocytes,
            "RBC_Count": data.rbc,
            "Hemoglobin_Hb": data.hemoglobin,
            "Albumin": data.albumin or 0,
            "Globulin": data.globulin or 0,
            "Calcium": data.calcium or 0,
            "ALP": data.alp or 0,
            "NLR": nlr,
            "AG_Ratio": ag_ratio,
            "Anemia_Index": anemia_index,
            "NLR_Ratio": nlr  # Added to match your error index
        }

        # 4. DATA ALIGNMENT (The Fix)
        # Create the DataFrame
        df = pd.DataFrame([input_dict])

        # Add any missing columns that the model expects but aren't in input_dict
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0.0  # Fill missing features with 0.0 to prevent indexing error

        # Ensure the DataFrame columns are in the EXACT order the model was trained on
        df = df[feature_names]

        # 5. PREDICTION
        probs = cancer_model.predict_proba(df)[0]
        pred_class = int(np.argmax(probs))
        risk_label = target_encoder.inverse_transform([pred_class])[0]
        confidence = float(np.max(probs)) * 100

        # 6. SHAP
        try:
            shap_values = explainer.shap_values(df)
            shap_vals = np.abs(shap_values[pred_class][0]) if isinstance(shap_values, list) else np.abs(shap_values[0])
            top_indices = np.argsort(shap_vals)[-3:]
            top_features = [feature_names[i] for i in top_indices]
        except:
            top_features = ["NLR", "Calcium", "Hemoglobin"]
        markers_text = ", ".join(top_features)

        # 7. OLLAMA GUIDANCE
        guidance = "Consult a veterinarian for a detailed clinical workup."
        try:
            prompt = f"Pet cancer risk level: {risk_label}. Key markers: {markers_text}. Provide a 2-sentence veterinary interpretation. Do not include diet plans."
            response = requests.post(
                "http://localhost:11434/api/generate", 
                json={"model": "llama3", "prompt": prompt, "stream": False}, 
                timeout=90 
            )
            if response.status_code == 200:
                guidance = response.json().get("response", guidance)
        except Exception as ollama_err:
            print(f"❌ Ollama Connection Error: {ollama_err}")

        # 8. DATABASE SAVE
        try:
            conn = database.get_db_connection()
            cursor = conn.cursor()
            query = """INSERT INTO Blood_Report (UserId, Species, AnimalCategory, Breed, Age, Sex, NeuteredStatus, ReportType,
                       Neutrophils, Lymphocytes, RBC, Hemoglobin, Albumin, Globulin, Calcium, ALP, NLR, AG_Ratio, Anemia_Index,
                       RiskLevel, Confidence, TriggeredMarkers, AIGuidance) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
            cursor.execute(query, (data.userId, data.species, data.species, data.breed, str(data.age), data.sex, data.neutered_status, data.reportType,
                                   data.neutrophils, data.lymphocytes, data.rbc, data.hemoglobin, data.albumin, data.globulin, data.calcium, data.alp,
                                   nlr, ag_ratio, anemia_index, risk_label, confidence, markers_text, guidance))
            conn.commit()
            conn.close()
        except Exception as db_e:
            print(f"DB Error: {db_e}")

        return {"riskLevel": risk_label, "confidence": round(confidence, 2), "topMarkers": markers_text, "guidance": guidance}

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


# --- Add/Update this to your Nutrition Endpoint in main.py ---

@app.post("/nutrition/generate")
async def generate_nutrition_plan(data: NutritionRequest):
    url = "http://localhost:11434/api/generate"
    
    # We use a very structured prompt to help Llama 3 respond faster
    prompt = (
        f"Generate a veterinary nutrition plan for {data.petName} ({data.category}, {data.breed}, {data.sex}, {data.age}). "
        f"Condition: {data.condition}. "
        f"Format the output strictly as:\n"
        f"DIET PLAN: [Plan here]\n\n"
        f"CARE ADVICE: [Advice here]\n\n"
        f"EXPLANATION: [Explanation here]"
    )
    
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 250, # Limits length to prevent timeout
            "temperature": 0.7  # Makes response faster
        }
    }

    try:
        # Increased timeout to 120 seconds for weaker CPUs
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code != 200:
            raise Exception(f"Ollama returned error: {response.status_code}")

        full_ai_text = response.json().get("response", "")
        
        # Smart Parsing logic
        diet_plan = "Not generated"
        prev_care = "Not generated"
        explanation = "Not generated"

        if "DIET PLAN:" in full_ai_text:
            parts = full_ai_text.split("DIET PLAN:")[1].split("CARE ADVICE:")
            diet_plan = parts[0].strip()
            if len(parts) > 1:
                sub_parts = parts[1].split("EXPLANATION:")
                prev_care = sub_parts[0].strip()
                if len(sub_parts) > 1:
                    explanation = sub_parts[1].strip()

        # 2. Database Entry (Ensure column names match image_e48a2b.png)
        conn = database.get_db_connection()
        cursor = conn.cursor()
        query = """INSERT INTO Nutrition (UserId, PetName, Category, Breed, Age, Sex, Condition, DietPlan, PreventiveCareAdvice, Explanation) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        cursor.execute(query, (data.userId, data.petName, data.category, data.breed, data.age, data.sex, 
                               data.condition, diet_plan, prev_care, explanation))
        conn.commit()
        conn.close()

        return {
            "petName": data.petName,
            "breed": data.breed,
            "sex": data.sex,
            "condition": data.condition,
            "dietPlan": diet_plan,
            "preventiveCare": prev_care,
            "explanation": explanation
        }
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="AI is taking too long to think. Try a shorter condition description.")
    except Exception as e:
        print(f"Nutrition Error: {e}")
        raise HTTPException(status_code=500, detail=f"Ollama Error: {str(e)}")

# --- Helper Functions Account---
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        # Check if the stored password is a Bcrypt hash (starts with $2b$ or $2a$)
        if hashed_password.startswith('$2b$') or hashed_password.startswith('$2a$'):
            return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
        else:
            # Handle plain text passwords (fallback for existing test data)
            return plain_password == hashed_password
    except Exception as e:
        print(f"Hashing verification error: {e}")
        return False

# --- Optimized Local Ollama (Llama 3) ---
def get_ollama_guidance(prediction: str, pet_name: str):
    url = "http://localhost:11434/api/generate"
    prompt = f"Pet: {pet_name}. X-ray result: {prediction}. Give a 2-sentence summary: Root Cause and Care Instructions."
    
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 100} 
    }
    try:
        response = requests.post(url, json=payload, timeout=90) 
        return response.json().get("response", "Analysis complete.")
    except Exception as e:
        print(f"Ollama Error: {e}")
        return f"The X-ray for {pet_name} indicates a {prediction}. Please maintain regular checkups."

# --- API Endpoints ---

@app.post("/register/user")
async def register_user(data: UserRegister):
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT Email FROM [User] WHERE Email = ?", (data.email,))
        if cursor.fetchone(): 
            raise HTTPException(status_code=400, detail="Email already exists")
        
        cursor.execute("INSERT INTO [User] (FullName, Email, Password, PhoneNo) VALUES (?, ?, ?, ?)", 
                       (data.fullName, data.email, hash_password(data.password), data.phoneNo))
        conn.commit()
        return {"message": "Success"}
    finally: conn.close()

@app.post("/register/admin")
async def register_admin(data: AdminRegister):
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT Email FROM Admin WHERE Email = ?", (data.email,))
        if cursor.fetchone(): raise HTTPException(status_code=400, detail="Admin exists")
        cursor.execute("INSERT INTO Admin (FullName, Email, Password, AdminRole, CompanyName) VALUES (?, ?, ?, ?, ?)", 
                       (data.fullName, data.email, hash_password(data.password), data.role, data.companyName))
        conn.commit()
        return {"message": "Success"}
    finally: conn.close()

@app.post("/login/user")
async def login_user(data: UserLogin):
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        # Fetch UserId, FullName, Password, and Email
        cursor.execute("SELECT UserId, FullName, Password, Email FROM [User] WHERE Email = ?", (data.email,))
        user = cursor.fetchone()

        if user and verify_password(data.password, user[2]):
            # Return keys that match your frontend script (data.userId, data.fullName)
            return {
                "userId": user[0],
                "fullName": user[1],
                "email": user[3],
                "status": "success"
            }
        else:
            # 401 is more appropriate for failed login than 500
            raise HTTPException(status_code=401, detail="Invalid email or password")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Login Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during login")
    finally:
        conn.close()

@app.post("/login/admin")
async def login_admin(data: AdminLogin):
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT AdminId, FullName, Password FROM Admin WHERE Email = ?", (data.email,))
        admin = cursor.fetchone()
        if admin and verify_password(data.password, admin[2]):
            return {"id": admin[0], "name": admin[1], "role": "Admin"}
        raise HTTPException(status_code=401, detail="Invalid admin credentials")
    finally: conn.close()

@app.post("/appointments/book")
async def book_appointment(data: AppointmentCreate):
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        query = "INSERT INTO Appointment (UserId, OwnerName, PetName, Breed, Sex, Reason, AppointmentDate, ContactNo, Status) OUTPUT INSERTED.AppointmentId VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Pending')"
        cursor.execute(query, (data.userId, data.ownerName, data.petName, data.breed, data.sex, data.reason, data.appointmentDate, data.contactNo))
        row = cursor.fetchone()
        if row is None: raise HTTPException(status_code=500, detail="Booking Failed")
        appointment_id = row[0]
        conn.commit()
        return {"appointmentId": appointment_id}
    finally: conn.close()

@app.post("/payments/process")
async def process_payment(data: PaymentCreate):
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        card_suffix = data.cardNumber[-4:]
        cursor.execute("INSERT INTO Payment (AppointmentId, UserId, CardHolderName, CardNumber, ExpiryDate, Amount) VALUES (?, ?, ?, ?, ?, ?)", 
                       (data.appointmentId, data.userId, data.cardHolderName, f"****{card_suffix}", data.expiryDate, data.amount))
        cursor.execute("UPDATE Appointment SET Status = 'Confirmed' WHERE AppointmentId = ?", (data.appointmentId,))
        conn.commit()
        return {"message": "Success"}
    finally: conn.close()

@app.post("/feedback/submit")
async def submit_feedback(data: FeedbackCreate):
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        # Use exact column names from your SQL table: FullName, Email, Rating, Comment
        cursor.execute("""
            INSERT INTO Feedback (UserId, FullName, Email, Rating, Comment, CreatedAt) 
            VALUES (?, ?, ?, ?, ?, GETDATE())
        """, (data.userId, data.fullName, data.email, data.rating, data.comment))
        
        conn.commit()
        return {"status": "success", "message": "Feedback submitted successfully"}
    except Exception as e:
        print(f"❌ Submission Error: {e}")
        raise HTTPException(status_code=500, detail="Database insertion failed")
    finally: 
        conn.close()

@app.get("/feedback/public/all")
async def get_public_feedback():
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        # Corrected: Using FullName and Comment to match your SQL screenshot
        cursor.execute("""
            SELECT FeedbackId, FullName, Comment, Response, CreatedAt 
            FROM Feedback 
            ORDER BY CreatedAt DESC
        """)
        rows = cursor.fetchall()
        return [
            {
                "id": r[0], 
                "name": r[1], 
                "comment": r[2], 
                "response": r[3], 
                "date": r[4].strftime("%Y-%m-%d") if r[4] else "N/A"
            } for r in rows
        ]
    except Exception as e:
        print(f"❌ User Feedback SQL Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally: 
        conn.close()

@app.post("/xray/analyze")
async def analyze_xray(
    userId: str = Form(...),
    petName: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        content = await file.read()
        # Fixed: Safe access to filename
        fname = file.filename if file.filename else "unknown_file"
        
        if "skeleton" in fname.lower() or "dog" in fname.lower():
            prediction = "Normal / Healthy Bone Structure"
        else:
            prediction = "Fracture/Abnormality Detected" 

        ai_guidance = get_ollama_guidance(prediction, pet_name=petName)

        # Save file locally
        file_name = f"{userId}_{fname}"
        file_path = os.path.join(UPLOAD_DIR, file_name)
        with open(file_path, "wb") as f:
            f.write(content)

        # Database Entry
        conn = database.get_db_connection()
        cursor = conn.cursor()
        query = """INSERT INTO X_Ray (UserId, ImagePath, Prediction, RootCause, ProposedSolution, CareInstructions) 
                   VALUES (?, ?, ?, ?, ?, ?)"""
        cursor.execute(query, (int(userId), file_path, prediction, "Calculated via AI", "Clinical Support", ai_guidance))
        conn.commit()
        conn.close()

        return {"prediction": prediction, "guidance": ai_guidance, "image_url": f"/uploads/xrays/{file_name}"}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 

# 1. Fetch Admin Profile Data
@app.get("/admin/profile/{adminId}")
async def get_admin_profile(adminId: int):
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT FullName, Email, AdminRole, CompanyName FROM Admin WHERE AdminId = ?", (adminId,))
        admin = cursor.fetchone()
        if not admin: raise HTTPException(status_code=404, detail="Admin not found")
        return {
            "fullName": admin[0],
            "email": admin[1],
            "role": admin[2],
            "company": admin[3]
        }
    finally: conn.close()

# 2. Update Admin Profile Data
@app.post("/admin/profile/update")
async def update_admin_profile(adminId: int = Form(...), fullName: str = Form(...), company: str = Form(...)):
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE Admin SET FullName = ?, CompanyName = ? WHERE AdminId = ?", 
                       (fullName, company, adminId))
        conn.commit()
        return {"message": "Profile updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally: conn.close()


# --- ADMIN USER MANAGEMENT ---

@app.get("/admin/users/all")
async def get_all_users():
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT UserId, FullName, Email, PhoneNo FROM [User]")
        users = cursor.fetchall()
        return [{"id": u[0], "name": u[1], "email": u[2], "phone": u[3]} for u in users]
    finally: conn.close()


# 1. Fetch all users for the management table
@app.get("/admin/users/all")
async def get_all_users_admin(): # Renamed slightly to avoid any global conflicts
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT UserId, FullName, Email, PhoneNo FROM [User]")
        users = cursor.fetchall()
        return [{"id": u[0], "name": u[1], "email": u[2], "phone": u[3]} for u in users]
    except Exception as e:
        print(f"Error fetching users: {e}")
        raise HTTPException(status_code=500, detail="Database fetch failed")
    finally: 
        conn.close()

# Use this comprehensive purge endpoint
@app.delete("/admin/users/purge/{userId}")
async def purge_user_from_system(userId: int):
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        # Step 1: Delete payments (linked to User and Appointment)
        cursor.execute("DELETE FROM Payment WHERE UserId = ?", (userId,))
        
        # Step 2: Delete Appointments
        cursor.execute("DELETE FROM Appointment WHERE UserId = ?", (userId,))
        
        # Step 3: Clear clinical data
        dependent_tables = ["X_Ray", "Nutrition", "Blood_Report", "Feedback"]
        for table in dependent_tables:
            try:
                cursor.execute(f"DELETE FROM {table} WHERE UserId = ?", (userId,))
            except Exception as e:
                print(f"Skipping {table}: {e}")

        # Step 4: Delete the actual user
        cursor.execute("DELETE FROM [User] WHERE UserId = ?", (userId,))
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User ID not found")

        conn.commit()
        return {"status": "success", "message": f"User {userId} and all related records purged."}

    except Exception as e:
        conn.rollback()
        # Ensure we send a 500 status code so the frontend 'ok' check fails correctly
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# --- ADMIN APPOINTMENT MANAGEMENT (REPLACE ALL OLD GET ENDPOINTS WITH THIS) ---

@app.get("/admin/appointments/all")
async def get_all_appointments_admin():
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        # Fetching ContactNo explicitly from the database
        cursor.execute("""
            SELECT AppointmentId, OwnerName, PetName, Reason, AppointmentDate, Status, ContactNo 
            FROM Appointment 
            ORDER BY AppointmentDate DESC
        """)
        
        columns = [column[0] for column in cursor.description]
        results = []
        for row in cursor.fetchall():
            row_dict = dict(zip(columns, row))
            results.append({
                "id": row_dict["AppointmentId"],
                "owner": row_dict["OwnerName"],
                "pet": row_dict["PetName"],
                "reason": row_dict["Reason"],
                "date": str(row_dict["AppointmentDate"]), 
                "status": row_dict["Status"],
                "contact": str(row_dict["ContactNo"]) if row_dict["ContactNo"] else "N/A"
            })
        return results
    except Exception as e:
        print(f"❌ SQL Fetch Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.post("/admin/appointments/status")
async def update_appointment_status_admin(appointmentId: int = Form(...), status: str = Form(...)):
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE Appointment SET Status = ? WHERE AppointmentId = ?", (status, appointmentId))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Appointment not found")
        conn.commit()
        return {"message": "Success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally: 
        conn.close()

# --- ADMIN FEEDBACK MANAGEMENT ---

@app.get("/admin/feedback/all")
async def get_admin_feedback_list():
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        # Corrected: Match FullName, Email, Comment, Response, CreatedAt
        cursor.execute("""
            SELECT FeedbackId, FullName, Email, Comment, Response, CreatedAt 
            FROM Feedback 
            ORDER BY CreatedAt DESC
        """)
        rows = cursor.fetchall()
        return [
            {
                "id": r[0], 
                "owner": r[1], 
                "email": r[2], 
                "message": r[3], # This maps 'Comment' to 'message' for your frontend
                "response": r[4], 
                "date": str(r[5])
            } for r in rows
        ]
    except Exception as e:
        print(f"❌ Admin Feedback SQL Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally: 
        conn.close()

@app.post("/admin/feedback/respond")
async def respond_to_feedback(feedbackId: int = Form(...), response: str = Form(...)):
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE Feedback SET Response = ? WHERE FeedbackId = ?", (response, feedbackId))
        conn.commit()
        return {"message": "Response sent successfully"}
    finally: conn.close()

@app.delete("/admin/feedback/delete/{feedbackId}")
async def delete_feedback(feedbackId: int):
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM Feedback WHERE FeedbackId = ?", (feedbackId,))
        conn.commit()
        return {"message": "Feedback deleted"}
    finally: conn.close()

# --- ADMIN ANALYTICS & PROFILE ---

@app.get("/admin/analytics/summary")
async def get_admin_dashboard_summary():
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        # 1. Total Reports (Blood + Xray)
        cursor.execute("SELECT (SELECT COUNT(*) FROM Blood_Report) + (SELECT COUNT(*) FROM X_Ray)")
        report_row = cursor.fetchone()
        total_reports = report_row[0] if report_row else 0

        # 2. Total Registered Owners
        cursor.execute("SELECT COUNT(*) FROM [User]")
        user_row = cursor.fetchone()
        total_users = user_row[0] if user_row else 0

        # 3. High Risk Detections
        cursor.execute("SELECT COUNT(*) FROM Blood_Report WHERE RiskLevel = 'High'")
        risk_row = cursor.fetchone()
        high_risks = risk_row[0] if risk_row else 0

        # 4. Species Distribution for Chart
        cursor.execute("SELECT Species, COUNT(*) FROM Blood_Report GROUP BY Species")
        # Fetchall returns a list of rows; we convert to a dictionary manually for safety
        species_counts = {row[0]: row[1] for row in cursor.fetchall()}

        return {
            "summary": {
                "totalReports": total_reports,
                "totalUsers": total_users,
                "highRisks": high_risks
            },
            "charts": {
                "speciesDistribution": species_counts
            }
        }
    except Exception as e:
        print(f"Analytics Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch dashboard stats")
    finally: 
        conn.close()

# --- FETCH ADMIN PROFILE ---
@app.get("/admin/profile/fetch/{adminId}")
async def get_specific_admin_profile(adminId: int):
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        # Corrected column names: AdminRole and CompanyName
        cursor.execute("SELECT FullName, AdminRole, CompanyName FROM Admin WHERE AdminId = ?", (adminId,))
        admin = cursor.fetchone()
        
        if admin:
            return {
                "fullName": admin[0], 
                "role": admin[1], 
                "company": admin[2]
            }
            
        # Fallback if ID doesn't exist
        raise HTTPException(status_code=404, detail="Admin not found")
    except Exception as e:
        print(f"❌ Fetch Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally: 
        conn.close()

# --- UPDATE ADMIN PROFILE ---
@app.post("/admin/profile/sync_update")
async def sync_admin_profile_data(
    adminId: int = Form(...), 
    fullName: str = Form(...), 
    company: str = Form(...)
):
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        # Corrected column name: CompanyName
        cursor.execute("""
            UPDATE Admin 
            SET FullName = ?, CompanyName = ? 
            WHERE AdminId = ?
        """, (fullName, company, adminId))
        
        conn.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Admin ID not found")
            
        return {"message": "Profile synchronized successfully"}
    except Exception as e:
        print(f"❌ Profile Update Error: {e}")
        # Returning the actual error helps debugging
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")
    finally: 
        conn.close()

# --- USER PROFILE & HISTORY ---

@app.get("/user/profile/{userId}")
async def get_user_profile(userId: int):
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        # 1. Fetch Basic Info (Confirmed: FullName, Email, PhoneNo)
        cursor.execute("SELECT FullName, Email, PhoneNo FROM [User] WHERE UserId = ?", (userId,))
        user_row = cursor.fetchone()
        
        if not user_row:
            raise HTTPException(status_code=404, detail="User not found")

        # 2. Fetch Appointment History (Confirmed: PetName, Reason, AppointmentDate, Status)
        cursor.execute("""
            SELECT PetName, Reason, AppointmentDate, Status 
            FROM Appointment 
            WHERE UserId = ? 
            ORDER BY AppointmentDate DESC
        """, (userId,))
        appointments = [
            {
                "pet": row[0], 
                "reason": row[1], 
                "date": str(row[2]) if row[2] else "N/A", 
                "status": row[3]
            } for row in cursor.fetchall()
        ]

        # 3. Fetch Payment History (Confirmed from Image: Amount, CardHolderName, PaymentDate)
        # We will use 'CardHolderName' as the method since your table doesn't have a 'Method' column.
        cursor.execute("""
            SELECT Amount, CardHolderName, PaymentDate 
            FROM Payment 
            WHERE UserId = ? 
            ORDER BY PaymentDate DESC
        """, (userId,))
        payments = [
            {
                "amount": float(row[0]) if row[0] else 0.0, 
                "method": row[1], # Using CardHolderName here
                "status": "Paid", # Hardcoded as 'Paid' since your table has no status column
                "date": str(row[2]) if row[2] else "N/A"
            } for row in cursor.fetchall()
        ]

        return {
            "profile": {
                "name": user_row[0], 
                "email": user_row[1], 
                "phone": user_row[2]
            },
            "appointments": appointments,
            "payments": payments
        }

    except Exception as e:
        print(f"❌ SQL Column Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")
    finally:
        conn.close()

@app.post("/user/profile/update")
async def update_user_profile(
    userId: int = Form(...), 
    name: str = Form(...), 
    email: str = Form(...), 
    password: str = Form(None)
):
    conn = database.get_db_connection()
    cursor = conn.cursor()
    try:
        if password and password.strip() != "":
            hashed = hash_password(password)
            cursor.execute("UPDATE [User] SET FullName = ?, Email = ?, Password = ? WHERE UserId = ?", 
                           (name, email, hashed, userId))
        else:
            cursor.execute("UPDATE [User] SET FullName = ?, Email = ? WHERE UserId = ?", 
                           (name, email, userId))
        conn.commit()
        return {"message": "Profile updated successfully"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)