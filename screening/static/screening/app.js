// Multi-step form logic for screening_form.html
(function () {
  let currentStepIndex = 0;
  let formSteps = [];
  let totalSteps = 0;

  function init() {
    formSteps = Array.from(document.querySelectorAll(".form-step"));
    totalSteps = formSteps.length;

    // update totalSteps display if exists
    const totalStepsEl = document.getElementById("totalSteps");
    if (totalStepsEl) totalStepsEl.textContent = totalSteps;

    console.log("screening app init, total steps =", totalSteps);
    showStep(0);
  }

  function showStep(index) {
    if (!formSteps.length) return;
    currentStepIndex = Math.max(0, Math.min(index, totalSteps - 1));

    formSteps.forEach((el, i) => {
      el.style.display = i === currentStepIndex ? "" : "none";
    });

    // prev/next button visibility and labels
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    if (prevBtn) prevBtn.style.display = currentStepIndex === 0 ? "none" : "";

    if (nextBtn) {
      if (currentStepIndex === totalSteps - 1) {
        nextBtn.textContent = "Proses Prediksi";
        nextBtn.type = "button";
      } else {
        nextBtn.textContent = "Selanjutnya";
        nextBtn.type = "button";
      }
    }

    // update progress
    const currentStepEl = document.getElementById("currentStep");
    if (currentStepEl) currentStepEl.textContent = currentStepIndex + 1;
    const progressFill = document.getElementById("progressFill");
    if (progressFill) {
      const pct = Math.round(((currentStepIndex + 1) / totalSteps) * 100);
      progressFill.style.width = pct + "%";
    }
    // If this is the final step and there's a summary area, populate it
    try {
      if (currentStepIndex === totalSteps - 1) {
        buildSummary();
      }
    } catch (e) {
      console.warn("buildSummary error", e);
    }
  }

  function validateStep(index) {
    const step = formSteps[index];
    if (!step) return true;

    const requiredEls = Array.from(step.querySelectorAll("[required]"));
    for (const el of requiredEls) {
      const val = el.value;
      if (el.tagName === "SELECT") {
        if (!val) {
          console.log("validation failed: empty select", el.name);
          el.focus();
          return false;
        }
      } else if (el.type === "checkbox" || el.type === "radio") {
        // if required and part of a group, check at least one checked
        const name = el.name;
        if (name) {
          const group = step.querySelectorAll('[name="' + name + '"]');
          const any = Array.from(group).some((g) => g.checked);
          if (!any) {
            console.log("validation failed: none checked in group", name);
            el.focus();
            return false;
          }
        }
      } else {
        if (val === null || val === undefined || String(val).trim() === "") {
          console.log("validation failed: empty input", el.name || el.tagName);
          el.focus();
          return false;
        }
      }
    }
    return true;
  }

  // Expose functions globally used by template onclick handlers
  window.nextStep = function () {
    console.log("nextStep called, currentStepIndex=", currentStepIndex);
    if (!formSteps.length) return;
    // if last step, submit
    if (currentStepIndex === totalSteps - 1) {
      const form = document.getElementById("screeningForm");
      if (form) {
        form.submit();
      }
      return;
    }

    // validate current step before moving on
    if (!validateStep(currentStepIndex)) {
      // simple feedback
      alert(
        "Mohon lengkapi semua field wajib pada halaman ini sebelum melanjutkan."
      );
      return;
    }

    showStep(currentStepIndex + 1);
  };

  window.previousStep = function () {
    if (!formSteps.length) return;
    showStep(currentStepIndex - 1);
  };

  function fmtBool(val) {
    if (val === null || val === undefined || val === "") return "-";
    const s = String(val);
    if (s === "1" || s.toLowerCase() === "true" || s.toLowerCase() === "ya")
      return "Ya";
    if (s === "0" || s.toLowerCase() === "false" || s.toLowerCase() === "tidak")
      return "Tidak";
    return s;
  }

  function buildSummary() {
    const form = document.getElementById("screeningForm");
    const container = document.getElementById("summaryContent");
    if (!form || !container) return;

    // Struktur kategori seperti app.js lama
    const categories = {
      "Informasi Dasar Pasien": [
        { key: "patient_name", label: "Nama Pasien" },
        { key: "district_city", label: "Kabupaten/Kota" },
        { key: "patient_age", label: "Umur (Tahun)", unit: "tahun" },
        { key: "education_level", label: "Pendidikan" },
        { key: "current_occupation", label: "Perkerjaan " },
        { key: "marital_status", label: "Status Nikah" },
        { key: "marriage_order", label: "Pernikahan Ke" },
        { key: "parity", label: "Paritas" },
      ],
      "Riwayat Kehamilan & Perencanaan": [
        {
          key: "new_partner_pregnancy",
          label: "Hamil Pasangan Baru",
          map: { 0: "Tidak", 1: "Ya" },
        },
        {
          key: "child_spacing_over_10_years",
          label: "Jarak Anak >10 tahun ",
          map: { 0: "Tidak", 1: "Ya" },
        },
        {
          key: "ivf_pregnancy",
          label: "Bayi Tabung ",
          map: { 0: "Tidak", 1: "Ya" },
        },
        {
          key: "multiple_pregnancy",
          label: "Gemelli",
          map: { 0: "Tidak", 1: "Ya" },
        },
        {
          key: "smoker",
          label: "Perokok ",
          map: { 0: "Tidak", 1: "Ya" },
        },
        {
          key: "planned_pregnancy",
          label: "Hamil Direncanakan ",
          map: { 0: "Tidak", 1: "Ya" },
        },
      ],
      "Riwayat Pribadi & Penyakit Ibu": [
        {
          key: "family_history_pe",
          label: "Riwayat Keluarga Preeklampsia",
          map: { 0: "Tidak", 1: "Ya" },
        },
        {
          key: "personal_history_pe",
          label: "Riwayat Preeklampsia",
          map: { 0: "Tidak", 1: "Ya" },
        },
        {
          key: "chronic_hypertension",
          label: "Hipertensi Kronis ",
          map: { 0: "Tidak", 1: "Ya" },
        },
        {
          key: "diabetes_mellitus",
          label: "Diabetes Melitus",
          map: { 0: "Tidak", 1: "Ya" },
        },
        {
          key: "kidney_disease",
          label: "Riwayat Penyakit Ginjal ",
          map: { 0: "Tidak", 1: "Ya" },
        },
        {
          key: "autoimmune_disease",
          label: "Penyakit Autoimune",
          map: { 0: "Tidak", 1: "Ya" },
        },
        {
          key: "aps_history",
          label: "APS",
          map: { 0: "Tidak", 1: "Ya" },
        },
      ],
      "Antropometri & Pemeriksaan": [
        {
          key: "pre_pregnancy_weight",
          label: "BB Sebelum Hamil (Kg)",
          unit: "kg",
        },
        { key: "height_cm", label: "TB (Cm)", unit: "cm" },
        { key: "bmi", label: "Indeks Massa Tubuh (IMT)", unit: "kg/m²" },
        { key: "lila_cm", label: "Lingkar Lengan Atas (Cm)", unit: "cm" },
        { key: "systolic_bp", label: "TD Sistolik I", unit: "mmHg" },
        { key: "diastolic_bp", label: "TD Diastolik I", unit: "mmHg" },
        { key: "map_mmhg", label: "MAP", unit: "mmHg" },
        { key: "hemoglobin", label: "Hb (gr/dl)", unit: "gr/dL" },
      ],
      "Riwayat Penyakit Keluarga": [
        {
          key: "family_history_hypertension",
          label: "Hipertensi Keluarga ",
          map: { 0: "Tidak", 1: "Ya" },
        },
        {
          key: "family_history_kidney",
          label: "Riwayat Penyakit Ginjal Keluarga",
          map: { 0: "Tidak", 1: "Ya" },
        },
        {
          key: "family_history_heart",
          label: "Riwayat Penyakit Jantung Keluarga ",
          map: { 0: "Tidak", 1: "Ya" },
        },
      ],
    };

    function getFieldValue(fieldDef) {
      const { key, map, unit } = fieldDef;
      const el = form.querySelector('[name="' + key + '"]');
      if (!el) return "-";

      let raw = "";

      if (el.tagName === "SELECT") {
        raw = el.value;
        // kalau ada map 0/1 → Ya/Tidak
        if (map) {
          const mKey = String(raw);
          if (map[mKey] !== undefined) return map[mKey];
        }
        // kalau tidak ada map, gunakan text-nya (SD, SMP, dst)
        if (!unit && !map) {
          return el.options[el.selectedIndex]?.text || raw || "-";
        }
      } else if (el.type === "checkbox" || el.type === "radio") {
        raw = el.checked ? el.value || "1" : "";
      } else {
        raw = el.value;
      }

      if (raw === undefined || raw === null || String(raw).trim() === "") {
        return "-";
      }

      // Kalau punya map tapi dari input non-select
      if (map) {
        const mKey = String(raw);
        if (map[mKey] !== undefined) return map[mKey];
        // fallback pakai fmtBool untuk 0/1
        return fmtBool(raw);
      }

      // Kalau ada unit (angka dengan satuan)
      if (unit) {
        return `${raw} ${unit}`;
      }

      // Default
      return raw;
    }

    let html = "";

    for (const [category, fields] of Object.entries(categories)) {
      html += `<div class="summary-section" style="margin-bottom: 20px;">`;
      html += `<h4 class="summary-category" style="font-size: 14px; font-weight: 600; margin-bottom: 10px; color: var(--primary, #0066ff);">${category}</h4>`;

      fields.forEach((field) => {
        const value = getFieldValue(field);

        html += `
        <div class="summary-item">
          <span class="summary-label">${field.label}</span>
          <span class="summary-value">${value}</span>
        </div>
      `;
      });

      html += `</div>`;
    }

    container.innerHTML = html;
  }

  // init when DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

// Dashboard filtering helpers: wired to elements in `dashboard.html`
window.loadDashboardData = function () {
  const search = document.getElementById("search-patient");
  if (search) {
    // Enter di input search -> filter
    search.addEventListener("keyup", function (e) {
      if (e.key === "Enter") {
        filterData();
      }
    });
  }

  const filterButtons = document.querySelectorAll(".btn-filter");
  filterButtons.forEach((btn) => {
    btn.addEventListener("click", function (e) {
      e.preventDefault();
      filterData();
    });
  });

  const clearButtons = document.querySelectorAll(".btn-clear");
  clearButtons.forEach((btn) => {
    btn.addEventListener("click", function (e) {
      e.preventDefault();
      clearFilters();
    });
  });

  // delegasi klik (boleh dibiarkan)
  document.addEventListener("click", function (ev) {
    const t = ev.target.closest && ev.target.closest(".btn-filter");
    if (t) {
      ev.preventDefault();
      filterData();
      return;
    }
    const c = ev.target.closest && ev.target.closest(".btn-clear");
    if (c) {
      ev.preventDefault();
      clearFilters();
      return;
    }
  });
};

window.filterData = function () {
  console.debug("filterData called");

  const searchTerm = (document.getElementById("search-patient")?.value || "")
    .trim()
    .toLowerCase();

  const resultFilter = (document.getElementById("filter-result")?.value || "")
    .trim()
    .toLowerCase(); // "" / "preeklampsia" / "non-preeklampsia"

  const tbody = document.getElementById("admin-table-body");
  if (!tbody) return;

  const rows = Array.from(tbody.querySelectorAll("tr"));

  rows.forEach((row) => {
    const cells = row.children;
    if (!cells.length) return;

    // Baris placeholder "Belum ada data prediksi"
    if (cells[0].hasAttribute("colspan")) {
      row.style.display = "";
      return;
    }

    // Struktur tabel:
    // 0: No
    // 1: Email
    // 2: Tanggal
    // 3: Nama Pasien
    // ...
    // n-1: Hasil (kolom terakhir)
    const nameCell = cells[3];
    const resultCell = cells[cells.length - 1];

    const nameText = (nameCell?.textContent || "").toLowerCase();
    const hasilTextRaw = (resultCell?.textContent || "").toLowerCase().trim();

    // Normalisasi hasil
    const hasilText = hasilTextRaw.replace(/\s+/g, " "); // "Non Preeklampsia" -> "non preeklampsia"

    let show = true;

    // === Filter berdasarkan nama pasien ===
    if (searchTerm && !nameText.includes(searchTerm)) {
      show = false;
    }

    // === Filter berdasarkan hasil ===
    if (show && resultFilter) {
      // Normalisasi hasilTextRaw untuk perbandingan yang tepat
      const normalizedHasil = hasilTextRaw
        .replace(/\s+/g, " ") // normalize spaces
        .replace(/-/g, " "); // replace hyphens with spaces
      
      const isPree =
        normalizedHasil.includes("preeklampsia") && 
        !normalizedHasil.includes("non");

      const isNonPree =
        normalizedHasil.includes("non") && 
        normalizedHasil.includes("preeklampsia");

      // resultFilter value: "preeklampsia" atau "non-preeklampsia"
      if (resultFilter === "preeklampsia" && !isPree) {
        show = false;
      }

      if (resultFilter === "non-preeklampsia" && !isNonPree) {
        show = false;
      }
    }

    row.style.display = show ? "" : "none";
  });
};

window.clearFilters = function () {
  console.debug("clearFilters called");
  const s = document.getElementById("search-patient");
  if (s) s.value = "";
  const rf = document.getElementById("filter-result");
  if (rf) rf.value = "";

  // tampilkan semua baris lagi
  const tbody = document.getElementById("admin-table-body");
  if (tbody) {
    Array.from(tbody.querySelectorAll("tr")).forEach((row) => {
      row.style.display = "";
    });
  }
};
