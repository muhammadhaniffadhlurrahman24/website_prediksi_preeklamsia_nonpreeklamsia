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
# Hasil prediksi HANYA berasal dari model ini
# =========================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),  # folder yang sama dengan views.py
    "ml_models",
    "rf_preeclampsia.joblib",  # MODEL UTAMA untuk prediksi
)

rf_model = None
try:
    # Load model Random Forest dari file rf_preeclampsia.joblib
    # Model ini adalah satu-satunya sumber hasil prediksi
    rf_model = joblib.load(MODEL_PATH)
    logger.info("RandomForest model loaded successfully from %s", MODEL_PATH)
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
    Default adalah 0.0 untuk menghindari NaN di model.
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
    if rf_model is not None:
        try:
            # Bentuk satu baris data sesuai kolom di training (ALL_FINAL.csv)
            # Map some categorical form values into numeric codes expected by the RF model
            # (training used numeric features; the RF here is not a preprocessing pipeline)
            education_map = {'SD': 0, 'SMP': 1, 'SMA': 2, 'D3': 3, 'S1': 4, 'S2': 5, 'S3': 6}
            occupation_map = {'Pedagang': 0, 'IRT': 1, 'Wiraswasta': 2, 'Swasta': 3, 'Guru': 4, 'Tani': 5}

            # Normalize / encode
            ed_raw = _clean_str(data.get('education_level'))
            edu_val = education_map.get(ed_raw, None)
            occ_raw = _clean_str(data.get('current_occupation'))
            occ_val = occupation_map.get(occ_raw, None)

            # Map parity to numeric: Primipara=1, Multipara=2, Grandemulti=5
            parity_map = {'Primipara': 1, 'Multipara': 2, 'Grandemulti': 5}
            parity_raw = _clean_str(data.get('parity'))
            parity_val = None
            if parity_raw:
                # Cek apakah sudah ada di mapping
                parity_val = parity_map.get(parity_raw)
                if parity_val is None:
                    # Fallback: try to extract numeric from string (untuk backward compatibility)
                    import re
                    m = re.search(r'(\d+)', parity_raw)
                    if m:
                        try:
                            parity_val = int(m.group(1))
                        except Exception:
                            parity_val = None

            # Map marital status to numeric: Sah=1 (menikah), Tidak=0, Siri=1 (juga menikah)
            marital_status_map = {'Sah': 1, 'Tidak': 0, 'Siri': 1}
            marital_raw = _clean_str(data.get('marital_status'))
            marital_val = marital_status_map.get(marital_raw, 0)  # Default 0 jika tidak ada

            # Helper function untuk mengkonversi boolean ke int (0/1) dengan default 0
            def _to_int(val, default=0):
                """Konversi boolean/None ke int, default 0 untuk menghindari NaN"""
                if val is None:
                    return default
                if isinstance(val, bool):
                    return 1 if val else 0
                if isinstance(val, (int, float)):
                    return int(val) if not (np.isnan(val) or np.isinf(val)) else default
                try:
                    val_str = str(val).strip().lower()
                    if val_str in ('1', 'true', 'ya', 'yes', 'on'):
                        return 1
                    return 0
                except Exception:
                    return default

            row = {
                # Urutan sesuai kolom training (29 fitur numerik)
                # Semua nilai harus numerik, tidak boleh None/NaN
                'Umur (Tahun)': _to_float(data.get('patient_age'), default=25.0),  # Default umur 25 tahun
                'Pendidikan': edu_val if edu_val is not None else 0,  # Default 0 jika tidak ada
                'Perkerjaan ': occ_val if occ_val is not None else 0,  # Default 0 jika tidak ada
                'Status Nikah': marital_val,  # Sudah default 0 di atas
                'Pernikahan Ke': _to_float(data.get('marriage_order'), default=1.0),  # Default 1
                'Paritas': parity_val if parity_val is not None else 0,  # Default 0 jika tidak ada
                'Hamil Pasangan Baru': _to_int(data.get('new_partner_pregnancy')),
                'Jarak Anak >10 tahun ': _to_int(data.get('child_spacing_over_10_years')),
                'Bayi Tabung ': _to_int(data.get('ivf_pregnancy')),
                'Gemelli': _to_int(data.get('multiple_pregnancy')),
                'Perokok ': _to_int(data.get('smoker')),
                'Hamil Direncanakan ': _to_int(data.get('planned_pregnancy')),
                'Riwayat Keluarga Preeklampsia': _to_int(data.get('family_history_pe')),
                'Riwayat Preeklampsia': _to_int(data.get('personal_history_pe')),
                'Hipertensi Kronis ': _to_int(data.get('chronic_hypertension')),
                'Diabetes Melitus': _to_int(data.get('diabetes_mellitus')),
                'Riwayat Penyakit Ginjal ': _to_int(data.get('kidney_disease')),
                'Penyakit Autoimune': _to_int(data.get('autoimmune_disease')),
                'APS': _to_int(data.get('aps_history')),
                'BB Sebelum Hamil (Kg)': _to_float(data.get('pre_pregnancy_weight'), default=55.0),  # Default 55 kg
                'TB (Cm)': _to_float(data.get('height_cm'), default=160.0),  # Default 160 cm
                'Indeks Massa Tubuh (IMT)': _to_float(data.get('bmi'), default=22.0),  # Default IMT 22
                'Lingkar Lengan Atas (Cm)': _to_float(data.get('lila_cm'), default=28.0),  # Default 28 cm
                'TD Sistolik I': _to_float(data.get('systolic_bp'), default=120.0),  # Default 120 mmHg
                'TD Diastolik I': _to_float(data.get('diastolic_bp'), default=80.0),  # Default 80 mmHg
                'MAP (mmHg)': _to_float(data.get('map_mmhg'), default=93.3),  # Default MAP ~93.3
                'Hb (gr/dl)': _to_float(data.get('hemoglobin'), default=12.0),  # Default 12 gr/dl
                'Hipertensi Keluarga ': _to_int(data.get('family_history_hypertension')),
                'Riwayat Penyakit Ginjal Keluarga': _to_int(data.get('family_history_kidney')),
            }

            X_input = pd.DataFrame([row])
            
            # Pastikan tidak ada NaN dalam DataFrame sebelum prediksi
            # Fill NaN dengan 0 sebagai fallback (seharusnya tidak terjadi karena sudah di-handle di atas)
            X_input = X_input.fillna(0)
            
            # Konversi semua kolom ke numeric, non-numeric akan menjadi NaN lalu di-fill dengan 0
            for col in X_input.columns:
                X_input[col] = pd.to_numeric(X_input[col], errors='coerce').fillna(0)

            # Convert ke numpy array tanpa nama kolom untuk menghindari warning
            # Model dilatih dengan array numpy tanpa feature names
            X_array = X_input.values

            # PREDIKSI UTAMA: Menggunakan model rf_preeclampsia.joblib
            # Hasil prediksi ditentukan oleh model Random Forest ini
            y_pred_raw = rf_model.predict(X_array)[0]

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
                    probas = rf_model.predict_proba(X_array)[0]
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
    }
    return render(request, "screening/result.html", context)


def result_view(request):
    # Simple render of result page (used when visiting /result/ directly)
    return render(request, "screening/result.html")


def download_result(request):
    from .models import ScreeningSubmission

    submission_id = request.GET.get("submission_id") or request.POST.get("submission_id")
    if submission_id:
        try:
            sub = ScreeningSubmission.objects.get(id=int(submission_id))
        except Exception:
            return HttpResponse("Submission not found", status=404)

        # Try to generate PDF using ReportLab; fall back to text
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas

            buffer = io.BytesIO()
            p = canvas.Canvas(buffer, pagesize=A4)
            width, height = A4
            y = height - 50
            line_height = 14

            lines = [
                f"LAPORAN PREDIKSI - ID: {sub.id}",
                "",
                f"Nama: {sub.patient_name}",
                f"Usia: {sub.patient_age}",
                f"Pendidikan: {sub.education_level}",
                f"IMT: {sub.bmi}",
                "",
                "Hasil Prediksi:",
                f" - {sub.result}",
                f" - Confidence: {sub.confidence}",
                "",
                f"Dibuat: {sub.created_at}",
            ]

            p.setFont("Helvetica", 11)
            for line in lines:
                if y < 50:
                    p.showPage()
                    p.setFont("Helvetica", 11)
                    y = height - 50
                p.drawString(40, y, str(line))
                y -= line_height

            p.showPage()
            p.save()
            buffer.seek(0)
            response = HttpResponse(buffer.getvalue(), content_type="application/pdf")
            response["Content-Disposition"] = f'attachment; filename="report_{sub.id}.pdf"'
            return response
        except ImportError:
            lines = [f"LAPORAN PREDIKSI - ID: {sub.id}", "", f"Nama: {sub.patient_name}", f"Usia: {sub.patient_age}", f"Pendidikan: {sub.education_level}", f"IMT: {sub.bmi}", "", "Hasil Prediksi:", f" - {sub.result}", f" - Confidence: {sub.confidence}", "", f"Dibuat: {sub.created_at}"]
            content = "\n".join(lines)
            response = HttpResponse(content, content_type="text/plain")
            response["Content-Disposition"] = f'attachment; filename="report_{sub.id}.txt"'
            return response

    # fallback
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
