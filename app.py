# app.py (with medical sanity-check overrides)
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import joblib, json, os
import pandas as pd
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
import faiss
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.secret_key = "super_secret_key"
MODEL_DIR = "models"

# load scaler & feature order
scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
feat_path = os.path.join(MODEL_DIR, "model_feature_order.json")
if not os.path.exists(scaler_path) or not os.path.exists(feat_path):
    raise FileNotFoundError("Missing trained artifacts. Run training notebook after generating strict dataset.")

scaler = joblib.load(scaler_path)
with open(feat_path, "r") as f:
    expected_features = json.load(f)

# load models (same naming as training notebook)
model_files = {}
for t in ["survived_1_year", "survived_5_years", "survived_10_years", "rejection_within_1yr",
          "quality_of_life_score", "rehospitalizations_in_1yr", "survival_time_months"]:
    if t == "survived_10_years":
        p = os.path.join(MODEL_DIR, "xgb_classifier_survival_10_years.joblib")
    elif "survived" in t or "rejection" in t:
        p = os.path.join(MODEL_DIR, f"xgb_classifier_{t}.joblib")
    else:
        p = os.path.join(MODEL_DIR, f"xgb_regressor_{t}.joblib")
    if os.path.exists(p):
        model_files[t] = joblib.load(p)
    else:
        model_files[t] = None

# load encoders
categorical_features = ['sex','donor_blood_type','recipient_blood_type','income_bracket','transplant_type']
encoders = {}
for f in categorical_features:
    fpath = os.path.join(MODEL_DIR, f"label_encoder_{f}.joblib")
    encoders[f] = joblib.load(fpath) if os.path.exists(fpath) else None

# user DB (same as before)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
USER_INDEX_PATH = os.path.join(DATA_DIR, "users.index")
USER_META_PATH = os.path.join(DATA_DIR, "users_meta.json")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
dim = 384
if os.path.exists(USER_INDEX_PATH) and os.path.exists(USER_META_PATH):
    try:
        user_index = faiss.read_index(USER_INDEX_PATH)
    except Exception:
        user_index = faiss.IndexFlatL2(dim)
    try:
        with open(USER_META_PATH,"r",encoding="utf-8") as f:
            user_metadata = json.load(f)
    except Exception:
        user_metadata = []
else:
    user_index = faiss.IndexFlatL2(dim)
    user_metadata = []

def get_vector(text):
    v = embedding_model.encode(text)
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    return v

def save_user_db():
    faiss.write_index(user_index, USER_INDEX_PATH)
    with open(USER_META_PATH, "w", encoding="utf-8") as f:
        json.dump(user_metadata, f, ensure_ascii=False, indent=2)

compat_rules = {'O': ['O','A','B','AB'],'A': ['A','AB'],'B': ['B','AB'],'AB': ['AB']}
def blood_compatible(donor, recipient):
    try:
        return int(str(recipient) in compat_rules.get(str(donor), []))
    except Exception:
        return 0

binary_map = {'Yes':1,'No':0,'yes':1,'no':0,1:1,0:0}

@app.route("/")
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("index.html")

# === LOGIN PAGE ===
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not email or not password:
            return render_template("login.html", error="Email and password required")

        if len(user_metadata) == 0 or user_index.ntotal == 0:
            return render_template("login.html", error="No users registered yet. Please sign up first.")

        vec = get_vector(email)
        try:
            distances, indices = user_index.search(vec, 1)
        except Exception as e:
            app.logger.error(f"FAISS search failed: {e}")
            return render_template("login.html", error="Internal search error")

        idx = int(indices[0][0])
        if idx < 0 or idx >= len(user_metadata):
            return render_template("login.html", error="No user found with these credentials.")

        user = user_metadata[idx]
        if user.get('email', '').lower() != email.lower():
            # Try to find the user manually
            found = None
            for u in user_metadata:
                if u.get('email', '').lower() == email.lower():
                    found = u
                    break
            if found is None:
                return render_template("login.html", error="No user found with these credentials.")
            user = found

        if check_password_hash(user['password_hash'], password):
            session['user'] = user['email']
            return redirect(url_for('index'))
        else:
            return render_template("login.html", error="No user found with these credentials.")

    return render_template("login.html")


# === SIGNUP PAGE ===
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not name or not email or not password:
            return render_template("signup.html", error="Name, email and password required")

        # Check for duplicate email (case-insensitive)
        for u in user_metadata:
            if u.get('email', '').lower() == email.lower():
                return render_template("signup.html", error="User already exists. Please login.")

        password_hash = generate_password_hash(password)
        vec = get_vector(email)

        try:
            user_index.add(vec)
            user_metadata.append({
                'name': name,
                'email': email,
                'password_hash': password_hash
            })

            os.makedirs(DATA_DIR, exist_ok=True)
            save_user_db()

            if not (os.path.exists(USER_INDEX_PATH) and os.path.exists(USER_META_PATH)):
                print("User files not created after save_user_db")
                return render_template("signup.html", error="Failed to save user data. Please check permissions.")

        except Exception as e:
            print(f"Exception during signup: {e}")
            app.logger.error(f"Failed to add user to index/meta: {e}")
            return render_template("signup.html", error=f"Failed to register user. Exception: {e}")

        return redirect(url_for('login'))

    return render_template("signup.html")


# === LOGOUT ===
@app.route("/logout")
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# === PREDICTION ENDPOINT ===
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        df_in = pd.DataFrame([data])

        # Derived features
        if 'bmi' not in df_in.columns and 'weight_kg' in df_in.columns and 'height_cm' in df_in.columns:
            try:
                df_in['bmi'] = float(df_in.at[0,'weight_kg']) / ((float(df_in.at[0,'height_cm'])/100)**2)
            except Exception:
                df_in['bmi'] = 0.0

        if 'donor_recipient_age_gap' not in df_in.columns and 'age' in df_in.columns and 'donor_age' in df_in.columns:
            try:
                df_in['donor_recipient_age_gap'] = abs(float(df_in.at[0,'age']) - float(df_in.at[0,'donor_age']))
            except Exception:
                df_in['donor_recipient_age_gap'] = 0.0

        if 'blood_type_compatible' not in df_in.columns and 'donor_blood_type' in df_in.columns and 'recipient_blood_type' in df_in.columns:
            df_in['blood_type_compatible'] = df_in.apply(lambda r: blood_compatible(r.get('donor_blood_type'), r.get('recipient_blood_type')), axis=1)

        # create *_codes columns expected by model if user passed raw categories
        for feat, le in encoders.items():
            code_col = feat + "_codes"
            if feat in df_in.columns:
                val = str(df_in.at[0, feat])
                if le is not None and val in list(le.classes_):
                    df_in[code_col] = int(le.transform([val])[0])
                else:
                    df_in[code_col] = 0
            elif code_col not in df_in.columns:
                df_in[code_col] = 0

        # map binary fields
        binary_fields = ['diabetes','hypertension','renal_dysfunction','prior_heart_surgery',
                         'smoking_history','alcohol_use','mechanical_support_before_tx','blood_type_compatible']
        for b in binary_fields:
            if b in df_in.columns:
                df_in[b] = df_in[b].map(binary_map).fillna(df_in[b])
            else:
                df_in[b] = 0

        # Ensure expected features exist
        for f in expected_features:
            if f not in df_in.columns:
                df_in[f] = 0.0

        # Ensure numeric
        df_in = df_in[expected_features].astype(float)
        X = scaler.transform(df_in)

        # Model predictions (if model artifact exists)
        preds = {}
        # classification outputs
        if model_files.get('survived_1_year') is not None:
            p1 = float(model_files['survived_1_year'].predict_proba(X)[0][1])
        else:
            p1 = None
        if model_files.get('survived_5_years') is not None:
            p5 = float(model_files['survived_5_years'].predict_proba(X)[0][1])
        else:
            p5 = None
        if model_files.get('rejection_within_1yr') is not None:
            pr = float(model_files['rejection_within_1yr'].predict_proba(X)[0][1])
        else:
            pr = None

        # regression outputs
        q = model_files['quality_of_life_score'].predict(X)[0] if model_files.get('quality_of_life_score') is not None else None
        reh = model_files['rehospitalizations_in_1yr'].predict(X)[0] if model_files.get('rehospitalizations_in_1yr') is not None else None
        surv_months = model_files['survival_time_months'].predict(X)[0] if model_files.get('survival_time_months') is not None else None

        # Ensure model_out is always defined
        model_out = {}
        if model_files.get('survived_1_year') is not None:
            p1 = float(model_files['survived_1_year'].predict_proba(X)[0][1])
        else:
            p1 = None
        if model_files.get('survived_5_years') is not None:
            p5 = float(model_files['survived_5_years'].predict_proba(X)[0][1])
        else:
            p5 = None
        if model_files.get('rejection_within_1yr') is not None:
            pr = float(model_files['rejection_within_1yr'].predict_proba(X)[0][1])
        else:
            pr = None
        q = model_files['quality_of_life_score'].predict(X)[0] if model_files.get('quality_of_life_score') is not None else None
        reh = model_files['rehospitalizations_in_1yr'].predict(X)[0] if model_files.get('rehospitalizations_in_1yr') is not None else None
        surv_months = model_files['survival_time_months'].predict(X)[0] if model_files.get('survival_time_months') is not None else None
        model_out = {
            'p1': p1, 'p5': p5, 'rej': pr, 'qol': q, 'rehosp': reh, 'survival_months': surv_months
        }

        # ---------------------------
        # MEDICAL SANITY CHECKS & OVERRIDES
        # ---------------------------
        # Read some raw input values (fallbacks)
        recipient_age = float(data.get('age', df_in.get('age', pd.Series([0])).iloc[0]))
        donor_bt = data.get('donor_blood_type', None)
        recipient_bt = data.get('recipient_blood_type', None)
        blood_compat = int(df_in.get('blood_type_compatible', pd.Series([0])).iloc[0])
        pra = int(float(data.get('pra_level', df_in.get('pra_level', pd.Series([0])).iloc[0])))
        age_gap = float(df_in.get('donor_recipient_age_gap', pd.Series([0])).iloc[0])

        # Rule 1: ABO incompatible (adult/child >=5)
        if blood_compat == 0 and recipient_age >= 5:
            # Override: virtually impossible survival without emergency measures
            model_out['p1'] = min(model_out['p1'] if model_out['p1'] is not None else 0.05, 0.05)
            model_out['p5'] = min(model_out['p5'] if model_out['p5'] is not None else 0.01, 0.01)
            model_out['rej'] = max(model_out['rej'] if model_out['rej'] is not None else 0.85, 0.85)
            # QoL & rehospitalizations adjustments
            if model_out['qol'] is not None:
                model_out['qol'] = max(1.0, model_out['qol'] - 3.0)
            if model_out['rehosp'] is not None:
                model_out['rehosp'] = model_out['rehosp'] + 3.0
            if model_out['survival_months'] is not None:
                model_out['survival_months'] = min(model_out['survival_months'], 6.0)

        # Rule 2: Very high PRA
        if pra >= 80:
            model_out['rej'] = max(model_out['rej'] if model_out['rej'] is not None else 0.8, 0.8)
            # decrease survival by 25-35 percentage points (bounded)
            if model_out['p1'] is not None:
                model_out['p1'] = max(0.01, model_out['p1'] - 0.30)
            if model_out['p5'] is not None:
                model_out['p5'] = max(0.005, model_out['p5'] - 0.40)
            if model_out['qol'] is not None:
                model_out['qol'] = max(1.0, model_out['qol'] - 1.5)
            if model_out['rehosp'] is not None:
                model_out['rehosp'] = model_out['rehosp'] + 2.0

        # Rule 3: Pediatric large age gap
        if (recipient_age < 18) and (age_gap > 30):
            # penalize survival and increase rejection
            if model_out['p1'] is not None:
                model_out['p1'] = max(0.01, model_out['p1'] - 0.30)
            if model_out['p5'] is not None:
                model_out['p5'] = max(0.005, model_out['p5'] - 0.40)
            model_out['rej'] = max(model_out['rej'] if model_out['rej'] is not None else 0.5, 0.6)
            if model_out['qol'] is not None:
                model_out['qol'] = max(1.0, model_out['qol'] - 1.0)
            if model_out['survival_months'] is not None:
                model_out['survival_months'] = max(3.0, model_out['survival_months'] * 0.5)

        # Rule 4: HLA mismatch + PRA synergy: if HLA high and PRA >50 raise rejection further
        try:
            hla = int(float(data.get('hla_mismatch_score', df_in.get('hla_mismatch_score', pd.Series([0])).iloc[0])))
            if (hla >= 4) and (pra >= 50):
                model_out['rej'] = max(model_out['rej'] if model_out['rej'] is not None else 0.6, 0.75)
                if model_out['p1'] is not None:
                    model_out['p1'] = max(0.01, model_out['p1'] - 0.20)
        except Exception:
            pass

        # Finally clamp numeric bounds and convert to human-readable percents/values
        def clamp(x, lo=0.0, hi=1.0):
            try:
                return float(max(lo, min(hi, x)))
            except Exception:
                return None
        # === RECOMPUTE p10 AFTER SANITY RULES ===
        import random
        if model_out.get('p5') is not None:
            p5 = max(0.0, min(1.0, model_out['p5']))

            # Optional: enforce p5 ≤ p1
            p1 = model_out.get('p1')
            if p1 is not None:
                p1 = max(0.0, min(1.0, p1))
                p5 = min(p5, p1)

            decay_exponent = random.uniform(1.9, 2.1)
            p10 = p5 ** decay_exponent

            if p10 >= p5:
                p10 = p5 - 0.001
            if p10 < 0:
                p10 = 0.0

            model_out['p10'] = round(p10, 4)
        else:
            model_out['p10'] = None

        out = {}
        if model_out['p1'] is not None:
            out['Chance of surviving 1 year (%)'] = round(clamp(model_out['p1'])*100, 2)
        if model_out['p5'] is not None:
            out['Chance of surviving 5 years (%)'] = round(clamp(model_out['p5'])*100, 2)
        if model_out.get('p10') is not None:
            out['Chance of surviving 10 years (%)'] = round(clamp(model_out['p10'])*100, 2)
        if model_out['rej'] is not None:
            out['Risk of rejection within 1 year (%)'] = round(clamp(model_out['rej'])*100, 2)
        if model_out['qol'] is not None:
            out['Predicted quality of life score'] = round(max(1.0, min(10.0, float(model_out['qol']))), 2)
        if model_out['rehosp'] is not None:
            out['Predicted rehospitalizations in 1 year'] = round(max(0.0, float(model_out['rehosp'])), 2)
        if model_out['survival_months'] is not None:
            out['Predicted survival time (months)'] = round(max(0.0, float(model_out['survival_months'])), 1)

        # Suggestion text (basic)
        suggestions = []
        if blood_compat == 0 and recipient_age >= 5:
            suggestions.append("ALERT: Donor → Recipient ABO incompatible for non-infant recipient — standard practice strongly contraindicates this transplant. Reevaluate donor selection or consider desensitization protocols and urgent specialist review.")
        if pra >= 80:
            suggestions.append("High PRA (>=80%) — high risk of antibody-mediated rejection. Strongly consider desensitization, plasmapheresis, or highly specialized immunomodulation; increase monitoring frequency.")
        if (recipient_age < 18) and (age_gap > 30):
            suggestions.append("Large donor-recipient age gap for pediatric recipient — increased risk. Multidisciplinary review recommended.")
        if model_out['rej'] is not None and model_out['rej'] >= 0.7:
            suggestions.append("High predicted rejection risk — consider more aggressive immunosuppression strategies and closer follow-up.")

        out['suggestions'] = suggestions

        return jsonify(out)
    
    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 400
    
if __name__ == "__main__":
    app.run(debug=True)
