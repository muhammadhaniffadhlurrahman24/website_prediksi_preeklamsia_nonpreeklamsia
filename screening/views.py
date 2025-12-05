from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout, get_user_model
from django.contrib import messages
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required, user_passes_test

import os
import json
import logging
import traceback

import joblib
import numpy as np
import pandas as pd
import io

from django.conf import settings

logger = logging.getLogger(__name__)

User = get_user_model()

# =========================
# LOAD MODEL RANDOM FOREST
# MODEL UTAMA: ml_models/rf_preeclampsia.joblib
# Model ini adalah Pipeline yang terdiri dari:
# - ColumnTransformer (preprocessing: imputation + onehot encoding)
# - RandomForestClassifier
# Hasil prediksi HANYA berasal dari model ini
# =========================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),  # folder yang sama dengan views.py
    "ml_models",
    "rf_preeclampsia.joblib",  # MODEL UTAMA untuk prediksi (Pipeline dengan preprocessing)
)

rf_model = None
try:
    # Load model Pipeline dari file rf_preeclampsia.joblib
    # Model ini sudah include preprocessing, jadi langsung predict dengan DataFrame
    rf_model = joblib.load(MODEL_PATH)
    logger.info("Model Pipeline loaded successfully from %s", MODEL_PATH)
    logger.info("Model type: %s", type(rf_model).__name__)
except FileNotFoundError:
    logger.error("Model file not found: %s", MODEL_PATH)
    logger.error("Prediksi tidak dapat dilakukan tanpa model rf_preeclampsia.joblib")
    rf_model = None
except Exception as e:
    logger.error("Failed to load RF model from %s: %s", MODEL_PATH, e)
    logger.error("Traceback: %s", traceback.format_exc())
    rf_model = None


# ==============
# VIEW DASAR
# ==============

def home(request):
    return redirect("login")


def login_view(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")

        user = authenticate(request, username=email, password=password)
        if user:
            login(request, user)
            messages.success(request, "Login berhasil.")
            return redirect("screening")

        return render(
            request,
            "screening/login.html",
            {"error": "Email atau password salah."},
        )

    return render(request, "screening/login.html")


def register_view(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        password = request.POST.get("password")
        confirm = request.POST.get("confirm_password")

        if password != confirm:
            return render(
                request,
                "screening/register.html",
                {"error": "Password dan konfirmasi tidak sama."},
            )

        if User.objects.filter(username=email).exists():
            return render(
                request,
                "screening/register.html",
                {"error": "User dengan email tersebut sudah ada."},
            )

        User.objects.create_user(
            username=email,
            email=email,
            password=password,
            first_name=name,
        )
        messages.success(request, "Registrasi berhasil. Silakan login.")
        return redirect("login")

    return render(request, "screening/register.html")


def admin_login_view(request):
    """
    Halaman login khusus admin.
    - Hanya menerima user dengan is_staff atau is_superuser.
    """
    if request.method == "POST":
        email = request.POST.get("email")
        # Support both 'admin_password' and 'password' input names (templates may vary)
        password = request.POST.get("admin_password") or request.POST.get("password")

        user = authenticate(request, username=email, password=password)
        if user and (user.is_staff or user.is_superuser):
            login(request, user)
            messages.success(request, "Login admin berhasil.")
            return redirect("admin_dashboard")

        return render(
            request,
            "screening/admin_login.html",
            {"error": "Email / password salah atau Anda bukan admin."},
        )

    return render(request, "screening/admin_login.html")


def admin_logout_view(request):
    logout(request)
    return redirect("admin_login")


def logout_view(request):
    logout(request)
    return redirect("login")


def screening_view(request):
    return render(request, "screening/screening_form.html")


# ===========================
# HELPER UNTUK KONVERSI DATA
# ===========================

def _to_float(val, default=0.0):
    """
    Ubah string/angka ke float, kalau gagal pakai default.
    Jika default=None, return None untuk missing values (akan di-handle oleh imputer).
    """
    if val is None or val == "":
        return default
    try:
        result = float(val)
        # Return default jika hasilnya NaN atau inf
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except Exception:
        return default


def _clean_str(val):
    """
    Bersihkan string dari spasi depan/belakang.
    """
    if val is None:
        return None
    return str(val).strip()


# ===========================
# SUBMIT SCREENING + PREDIKSI
# ===========================

def submit_screening(request):
    from .models import ScreeningSubmission  # import lokal

    if request.method != "POST":
        return redirect("screening")

    def to_bool(val):
        """
        Form kirim "0"/"1". Di sini diubah ke bool atau None.
        Bool akan diperlakukan sebagai 0/1 oleh OneHotEncoder/RandomForest.
        """
        if val in (None, ""):
            return None
        return True if str(val) in ("1", "True", "true", "on") else False

    # Kumpulkan data mentah dari form
    data = {
        "patient_name": request.POST.get("patient_name", ""),
        "district_city": request.POST.get("district_city", ""),
        "patient_age": request.POST.get("patient_age") or None,
        "education_level": request.POST.get("education_level", ""),
        "current_occupation": request.POST.get("current_occupation", ""),
        "marital_status": request.POST.get("marital_status", ""),
        "marriage_order": request.POST.get("marriage_order") or None,
        "parity": request.POST.get("parity", ""),

        "new_partner_pregnancy": to_bool(request.POST.get("new_partner_pregnancy")),
        "child_spacing_over_10_years": to_bool(request.POST.get("child_spacing_over_10_years")),
        "ivf_pregnancy": to_bool(request.POST.get("ivf_pregnancy")),
        "multiple_pregnancy": to_bool(request.POST.get("multiple_pregnancy")),
        "smoker": to_bool(request.POST.get("smoker")),
        "planned_pregnancy": to_bool(request.POST.get("planned_pregnancy")),

        "family_history_pe": to_bool(request.POST.get("family_history_pe")),
        "personal_history_pe": to_bool(request.POST.get("personal_history_pe")),
        "chronic_hypertension": to_bool(request.POST.get("chronic_hypertension")),
        "diabetes_mellitus": to_bool(request.POST.get("diabetes_mellitus")),
        "kidney_disease": to_bool(request.POST.get("kidney_disease")),
        "autoimmune_disease": to_bool(request.POST.get("autoimmune_disease")),
        "aps_history": to_bool(request.POST.get("aps_history")),

        "pre_pregnancy_weight": request.POST.get("pre_pregnancy_weight") or None,
        "height_cm": request.POST.get("height_cm") or None,
        "bmi": request.POST.get("bmi") or None,
        "lila_cm": request.POST.get("lila_cm") or None,
        "systolic_bp": request.POST.get("systolic_bp") or None,
        "diastolic_bp": request.POST.get("diastolic_bp") or None,
        "map_mmhg": request.POST.get("map_mmhg") or None,
        "hemoglobin": request.POST.get("hemoglobin") or None,

        "family_history_hypertension": to_bool(request.POST.get("family_history_hypertension")),
        "family_history_kidney": to_bool(request.POST.get("family_history_kidney")),
        "family_history_heart": to_bool(request.POST.get("family_history_heart")),
    }

    # Konversi angka ke int/float
    for key in ("patient_age", "marriage_order", "systolic_bp", "diastolic_bp"):
        if data.get(key) is not None:
            try:
                data[key] = int(data[key])
            except Exception:
                data[key] = None

    for key in (
        "pre_pregnancy_weight",
        "height_cm",
        'bmi',
        'lila_cm',
        'map_mmhg',
        'hemoglobin',
    ):
        if data.get(key) is not None:
            try:
                data[key] = float(data[key])
            except Exception:
                data[key] = None

    # Validasi minimal
    if not data.get("patient_name") or data.get("patient_age") is None:
        form_data_json = json.dumps(request.POST.dict())
        return render(
            request,
            "screening/screening_form.html",
            {
                "error": "Nama pasien dan umur wajib diisi.",
                "form_data": form_data_json,
            },
        )

    # ==============================
    # PREDIKSI MENGGUNAKAN MODEL RF
    # HASIL PREDIKSI HANYA BERASAL DARI: ml_models/rf_preeclampsia.joblib
    # ==============================

    is_pree = False
    conf_val = 0.0

    # Validasi: Model HARUS tersedia untuk melakukan prediksi
    if rf_model is None:
        logger.error("Model rf_preeclampsia.joblib tidak tersedia! Prediksi tidak dapat dilakukan.")
        form_data_json = json.dumps(request.POST.dict())
        return render(
            request,
            "screening/screening_form.html",
            {
                "error": "Sistem prediksi sedang tidak tersedia. Silakan hubungi administrator.",
                "form_data": form_data_json,
            },
        )

    # Prediksi HANYA menggunakan model rf_preeclampsia.joblib
    # Model ini adalah Pipeline dengan ColumnTransformer yang melakukan preprocessing otomatis
    if rf_model is not None:
        try:
            # Helper function untuk mengkonversi boolean ke string Ya/Tidak
            def _to_yesno(val):
                """Konversi boolean/None ke string Ya/Tidak untuk categorical features"""
                if val is None:
                    return "Tidak"
                if isinstance(val, bool):
                    return "Ya" if val else "Tidak"
                if isinstance(val, (int, float)):
                    if np.isnan(val) or np.isinf(val):
                        return "Tidak"
                    return "Ya" if int(val) == 1 else "Tidak"
                try:
                    val_str = str(val).strip().lower()
                    if val_str in ('1', 'true', 'ya', 'yes', 'on'):
                        return "Ya"
                    return "Tidak"
                except Exception:
                    return "Tidak"

            # Bentuk DataFrame dengan nama kolom yang sama persis dengan training
            # Model pipeline akan otomatis melakukan preprocessing (imputation + onehot encoding)
            row = {
                # Numeric features (akan di-impute dengan median jika missing)
                'Umur (Tahun)': _to_float(data.get('patient_age'), default=None),
                'Pernikahan Ke': _to_float(data.get('marriage_order'), default=None),
                'BB Sebelum Hamil (Kg)': _to_float(data.get('pre_pregnancy_weight'), default=None),
                'TB (Cm)': _to_float(data.get('height_cm'), default=None),
                'Indeks Massa Tubuh (IMT)': _to_float(data.get('bmi'), default=None),
                'Lingkar Lengan Atas (Cm)': _to_float(data.get('lila_cm'), default=None),
                'TD Sistolik I': _to_float(data.get('systolic_bp'), default=None),
                'TD Diastolik I': _to_float(data.get('diastolic_bp'), default=None),
                'MAP (mmHg)': _to_float(data.get('map_mmhg'), default=None),
                'Hb (gr/dl)': _to_float(data.get('hemoglobin'), default=None),
                
                # Categorical features (akan di-impute dengan most_frequent + onehot encoded)
                'Kabupaten/Kota': _clean_str(data.get('district_city')) or None,
                'Pendidikan': _clean_str(data.get('education_level')) or None,
                'Pekerjaan': _clean_str(data.get('current_occupation')) or None,
                'Status Nikah': _clean_str(data.get('marital_status')) or None,
                'Paritas': _clean_str(data.get('parity')) or None,
                'Hamil Pasangan Baru': _to_yesno(data.get('new_partner_pregnancy')),
                'Jarak Anak >10 tahun': _to_yesno(data.get('child_spacing_over_10_years')),
                'Bayi Tabung': _to_yesno(data.get('ivf_pregnancy')),
                'Gemelli': _to_yesno(data.get('multiple_pregnancy')),
                'Perokok': _to_yesno(data.get('smoker')),
                'Hamil Direncanakan': _to_yesno(data.get('planned_pregnancy')),
                'Riwayat Keluarga Preeklampsia': _to_yesno(data.get('family_history_pe')),
                'Riwayat Preeklampsia': _to_yesno(data.get('personal_history_pe')),
                'Hipertensi Kronis': _to_yesno(data.get('chronic_hypertension')),
                'Diabetes Melitus': _to_yesno(data.get('diabetes_mellitus')),
                'Riwayat Penyakit Ginjal': _to_yesno(data.get('kidney_disease')),
                'Penyakit Autoimune': _to_yesno(data.get('autoimmune_disease')),
                'APS': _to_yesno(data.get('aps_history')),
                'Hipertensi Keluarga': _to_yesno(data.get('family_history_hypertension')),
                'Riwayat Penyakit Ginjal Keluarga': _to_yesno(data.get('family_history_kidney')),
                'Riwayat Penyakit Jantung Keluarga': _to_yesno(data.get('family_history_heart')),
            }

            # Buat DataFrame dengan nama kolom yang sama persis dengan training
            X_input = pd.DataFrame([row])
            
            # Model pipeline akan otomatis melakukan:
            # 1. Imputation (median untuk numeric, most_frequent untuk categorical)
            # 2. OneHotEncoding untuk categorical features
            # 3. Prediksi dengan RandomForest

            # PREDIKSI UTAMA: Menggunakan model pipeline rf_preeclampsia.joblib
            # Model ini sudah include preprocessing, jadi langsung predict dengan DataFrame
            y_pred_raw = rf_model.predict(X_input)[0]

            # Normalize predicted label to canonical string values used elsewhere
            # ('Preeklampsia' or 'NonPreeklampsia')
            def _normalize_label(raw):
                try:
                    # numeric types -> 1 means Preeklampsia
                    if isinstance(raw, (int, float, np.integer)):
                        return 'Preeklampsia' if int(raw) == 1 else 'NonPreeklampsia'
                    s = str(raw).strip()
                    key = s.lower().replace('-', '').replace(' ', '')
                    if key in ('preeklampsia', 'preeclampsia'):
                        return 'Preeklampsia'
                    if key in ('nonpreeklampsia', 'nonpreeclampsia', 'nonpreeklampsia'):
                        return 'NonPreeklampsia'
                    # if it's a numeric string
                    try:
                        if int(s) == 1:
                            return 'Preeklampsia'
                        else:
                            return 'NonPreeklampsia'
                    except Exception:
                        return s
                except Exception:
                    return str(raw)

            y_pred = _normalize_label(y_pred_raw)

            # Confidence calculation: try to use predict_proba and locate the Preeklampsia class index
            if hasattr(rf_model, 'predict_proba'):
                try:
                    probas = rf_model.predict_proba(X_input)[0]
                    classes = list(rf_model.classes_)

                    # find index corresponding to Preeklampsia
                    idx_pree = None
                    for idx, c in enumerate(classes):
                        try:
                            if isinstance(c, (int, float, np.integer)) and int(c) == 1:
                                idx_pree = idx
                                break
                            kc = str(c).lower().replace('-', '').replace(' ', '')
                            if kc in ('preeklampsia', 'preeclampsia'):
                                idx_pree = idx
                                break
                        except Exception:
                            continue

                    if idx_pree is not None:
                        pree_proba = float(probas[idx_pree]) * 100.0
                        conf_val = pree_proba if y_pred == 'Preeklampsia' else 100.0 - pree_proba
                    else:
                        conf_val = float(max(probas) * 100.0)
                except Exception:
                    conf_val = 75.0
            else:
                conf_val = 75.0

            is_pree = (y_pred == 'Preeklampsia')

        except Exception as e:
            # Jika terjadi error saat prediksi dengan model, log error dan return error
            logger.error("Error during RF prediction using rf_preeclampsia.joblib: %s", e)
            logger.error("Traceback: %s", traceback.format_exc())
            form_data_json = json.dumps(request.POST.dict())
            return render(
                request,
                "screening/screening_form.html",
                {
                    "error": f"Terjadi kesalahan saat melakukan prediksi: {str(e)}. Silakan coba lagi atau hubungi administrator.",
                    "form_data": form_data_json,
                },
            )

    data["result"] = "Preeklampsia" if is_pree else "Non-Preeklampsia"
    data["confidence"] = f"{conf_val:.1f}%"

    # Attach user jika login
    if request.user.is_authenticated:
        data["user"] = request.user

    # Simpan ke database
    try:
        submission = ScreeningSubmission.objects.create(**data)
    except Exception:
        logger.exception("Failed to save ScreeningSubmission")
        form_data_json = json.dumps(request.POST.dict())
        return render(
            request,
            "screening/screening_form.html",
            {
                "error": "Terjadi kesalahan saat menyimpan data. Silakan coba lagi.",
                "form_data": form_data_json,
            },
        )

    # Rekomendasi sederhana
    recommendations = [
        "Konsultasikan ke dokter kandungan.",
        "Kontrol tekanan darah secara rutin.",
    ]
    if is_pree:
        recommendations.insert(0, "Segera lakukan evaluasi klinis lebih lanjut.")

    context = {
        "result": data["result"],
        "confidence": data["confidence"],
        "patient_name": data.get("patient_name"),
        "patient_age": data.get("patient_age"),
        "education": data.get("education_level"),
        "bmi": data.get("bmi"),
        "recommendations": recommendations,
        "submission_id": submission.id,
        "submission": submission,  # Tambahkan submission object untuk akses semua field
    }
    return render(request, "screening/result.html", context)


def result_view(request):
    # Simple render of result page (used when visiting /result/ directly)
    return render(request, "screening/result.html")


def download_result(request):
    from .models import ScreeningSubmission
    from django.utils import timezone

    submission_id = request.GET.get("submission_id") or request.POST.get("submission_id")
    preview = request.GET.get("preview")

    if submission_id:
        try:
            sub = ScreeningSubmission.objects.get(id=int(submission_id))
        except Exception:
            return HttpResponse("Submission not found", status=404)

        def yes_no(val):
            if val is True or val == "True" or str(val) == "1":
                return "Ya"
            if val is False or val == "False" or str(val) == "0":
                return "Tidak"
            return "-"

        # Hitung IMT (sama seperti preview)
        bmi_val = "-"
        if sub.bmi:
            try:
                bmi_val = f"{float(sub.bmi):.1f}"
            except:
                pass
        elif sub.pre_pregnancy_weight and sub.height_cm:
            try:
                bb = float(sub.pre_pregnancy_weight)
                tb = float(sub.height_cm)
                if tb > 0:
                    bmi_calc = bb / ((tb / 100) ** 2)
                    bmi_val = f"{bmi_calc:.1f}"
            except:
                pass

        # Format tanggal sama seperti preview (toLocaleDateString("id-ID"))
        if sub.created_at:
            tanggal = sub.created_at.strftime("%d/%m/%Y")
        else:
            tanggal = timezone.now().strftime("%d/%m/%Y")
        
        # Format prediksi (sama seperti preview)
        result_lower = (sub.result or "").lower().replace("-", "").replace(" ", "")
        is_pree = result_lower in ("preeklampsia", "preeclampsia")
        prediksi_text = "PREEKLAMPSIA" if is_pree else "NON-PREEKLAMPSIA"
        
        # Format confidence (pastikan ada % jika belum ada)
        confidence_text = sub.confidence or "-"
        if confidence_text != "-" and "%" not in str(confidence_text):
            confidence_text = f"{confidence_text}%"

        # Buat HTML dengan format yang sama persis seperti preview
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Laporan Prediksi Preeklampsia</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #0066FF; }}
        .result-box {{ padding: 20px; background: #f0f9ff; border: 2px solid #0066FF; border-radius: 8px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        td {{ padding: 8px 10px; border-bottom: 1px solid #ccc; vertical-align: top; }}
        .label {{ font-weight: bold; width: 35%; }}
        .recommendations {{ margin-top: 30px; }}
        .recommendations ul {{ margin-top: 10px; }}
        .recommendations li {{ margin-bottom: 8px; }}
        h3 {{ margin-top: 30px; color: #111827; }}
    </style>
</head>
<body>
    <h1>Laporan Hasil Prediksi Preeklampsia</h1>
    <p>Tanggal: {tanggal}</p>
    
    <div class="result-box">
        <h2>{prediksi_text}</h2>
        <p>Kepercayaan: <strong>{confidence_text}</strong></p>
    </div>
    
    <h3>Data Pasien</h3>
    <table>
        <tr><td class="label">Nama Pasien</td><td>{sub.patient_name or '-'}</td></tr>
        <tr><td class="label">Umur Pasien</td><td>{f'{sub.patient_age} tahun' if sub.patient_age else '-'}</td></tr>
        <tr><td class="label">Status Pendidikan Terakhir</td><td>{sub.education_level or '-'}</td></tr>
        <tr><td class="label">Pekerjaan Saat Ini</td><td>{sub.current_occupation or '-'}</td></tr>
        <tr><td class="label">Pernikahan Ke</td><td>{sub.marriage_order or '-'}</td></tr>
        <tr><td class="label">Paritas</td><td>{sub.parity or '-'}</td></tr>
    </table>
    <h3>Riwayat Kehamilan & Perencanaan</h3>
    <table>
        <tr><td class="label">Hamil Pasangan Baru</td><td>{yes_no(sub.new_partner_pregnancy)}</td></tr>
        <tr><td class="label">Jarak Anak > 10 Tahun</td><td>{yes_no(sub.child_spacing_over_10_years)}</td></tr>
        <tr><td class="label">Bayi Tabung</td><td>{yes_no(sub.ivf_pregnancy)}</td></tr>
        <tr><td class="label">Gemeli (Kehamilan Kembar)</td><td>{yes_no(sub.multiple_pregnancy)}</td></tr>
        <tr><td class="label">Perokok</td><td>{yes_no(sub.smoker)}</td></tr>
        <tr><td class="label">Hamil Direncanakan</td><td>{yes_no(sub.planned_pregnancy)}</td></tr>
    </table>
    <h3>Riwayat Pribadi & Penyakit Ibu</h3>
    <table>
        <tr><td class="label">Riwayat Keluarga PE</td><td>{yes_no(sub.family_history_pe)}</td></tr>
        <tr><td class="label">Riwayat PE</td><td>{yes_no(sub.personal_history_pe)}</td></tr>
        <tr><td class="label">HT Kronis</td><td>{yes_no(sub.chronic_hypertension)}</td></tr>
        <tr><td class="label">DM</td><td>{yes_no(sub.diabetes_mellitus)}</td></tr>
        <tr><td class="label">Penyakit Ginjal</td><td>{yes_no(sub.kidney_disease)}</td></tr>
        <tr><td class="label">Autoimune</td><td>{yes_no(sub.autoimmune_disease)}</td></tr>
        <tr><td class="label">APS</td><td>{yes_no(sub.aps_history)}</td></tr>
    </table>
    <h3>Antropometri & Pemeriksaan</h3>
    <table>
        <tr><td class="label">BB Sebelum Hamil</td><td>{f'{sub.pre_pregnancy_weight} kg' if sub.pre_pregnancy_weight else '-'}</td></tr>
        <tr><td class="label">Tinggi Badan</td><td>{f'{sub.height_cm} cm' if sub.height_cm else '-'}</td></tr>
        <tr><td class="label">IMT</td><td>{f'{bmi_val} kg/m²' if bmi_val != '-' else '-'}</td></tr>
        <tr><td class="label">LiLA</td><td>{f'{sub.lila_cm} cm' if sub.lila_cm else '-'}</td></tr>
        <tr><td class="label">TD Sistolik</td><td>{f'{sub.systolic_bp} mmHg' if sub.systolic_bp else '-'}</td></tr>
        <tr><td class="label">TD Diastolik</td><td>{f'{sub.diastolic_bp} mmHg' if sub.diastolic_bp else '-'}</td></tr>
        <tr><td class="label">MAP</td><td>{f'{sub.map_mmhg} mmHg' if sub.map_mmhg else '-'}</td></tr>
        <tr><td class="label">Hb</td><td>{f'{sub.hemoglobin} gr/dL' if sub.hemoglobin else '-'}</td></tr>
    </table>
    <h3>Riwayat Penyakit Keluarga</h3>
    <table>
        <tr><td class="label">HT Keluarga</td><td>{yes_no(sub.family_history_hypertension)}</td></tr>
        <tr><td class="label">Ginjal Keluarga</td><td>{yes_no(sub.family_history_kidney)}</td></tr>
        <tr><td class="label">Jantung Keluarga</td><td>{yes_no(sub.family_history_heart)}</td></tr>
    </table>
    
    <div class="recommendations">
        <h3>Rekomendasi</h3>
        <ul>
            {f'<li>Segera lakukan evaluasi klinis lebih lanjut.</li>' if is_pree else ''}
            <li>Konsultasikan ke dokter kandungan.</li>
            <li>Kontrol tekanan darah secara rutin.</li>
        </ul>
    </div>
    
    <p style="margin-top: 40px; font-size: 12px; color: #666;">
        <strong>Disclaimer:</strong> Laporan ini adalah hasil analisis otomatis dan bukan pengganti konsultasi medis profesional. 
        Selalu konsultasikan dengan dokter atau tenaga medis profesional untuk diagnosis dan pengobatan yang tepat.
    </p>
</body>
</html>
        """

        # Try to convert HTML to PDF using xhtml2pdf
        try:
            from xhtml2pdf import pisa
            
            buffer = io.BytesIO()
            pisa_status = pisa.CreatePDF(html_content, dest=buffer, encoding='utf-8')
            
            if pisa_status.err:
                # Jika xhtml2pdf gagal, return HTML untuk preview/print
                response = HttpResponse(html_content, content_type="text/html")
                if preview:
                    return response
                response["Content-Disposition"] = f'inline; filename="report_{sub.id}.html"'
                return response
            
            buffer.seek(0)
            response = HttpResponse(buffer.getvalue(), content_type="application/pdf")
            disp_type = "inline" if preview else "attachment"
            response["Content-Disposition"] = f'{disp_type}; filename="report_{sub.id}.pdf"'
            return response
        except ImportError:
            # Jika xhtml2pdf tidak terpasang, return HTML untuk preview/print
            response = HttpResponse(html_content, content_type="text/html")
            if preview:
                return response
            response["Content-Disposition"] = f'inline; filename="report_{sub.id}.html"'
            return response

    # fallback tanpa submission_id
    content = "Laporan prediksi sederhana\nGunakan fitur ini untuk men-generate laporan nyata.\n"
    response = HttpResponse(content, content_type="text/plain")
    response["Content-Disposition"] = 'attachment; filename="report.txt"'
    return response


@login_required
def my_submissions(request):
    from .models import ScreeningSubmission

    subs = ScreeningSubmission.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "screening/my_submissions.html", {"submissions": subs})


def admin_dashboard(request):
    from .models import ScreeningSubmission

    total_predictions = ScreeningSubmission.objects.count()
    # Hitung persis sesuai label yang disimpan aplikasi
    preeclampsia_count = ScreeningSubmission.objects.filter(result__iexact="Preeklampsia").count()
    non_preeclampsia_count = ScreeningSubmission.objects.filter(result__iexact="Non-Preeklampsia").count()
    submissions = ScreeningSubmission.objects.select_related("user").order_by("-created_at")[:200]

    context = {
        "total_predictions": total_predictions,
        "preeclampsia_count": preeclampsia_count,
        "non_preeclampsia_count": non_preeclampsia_count,
        "total_users": User.objects.count(),
        "submissions": submissions,
    }
    return render(request, "screening/dashboard.html", context)
