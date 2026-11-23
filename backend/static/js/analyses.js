/*
 * analyses.js — 분석 화면 · 상세 화면 동작
 *
 * 분석 화면 흐름 (버튼 1번 클릭)
 *   1) POST /api/analyses/  (FormData: 영상 또는 카메라 주소)  → 202 + 작업 ID 즉시 반환
 *   2) GET  /api/analyses/{id}/status/ 를 1초마다 조회 → 단계(대기 → 영상 수집 → AI 분석 → 완료) 표시
 *   3) 완료되면 사람별 결과 표 표시, 최근 이력 맨 위에 추가
 */
(() => {
  "use strict";

  // ── 공통 ───────────────────────────────────────────────
  const getCookie = (name) =>
    document.cookie.split("; ").find((c) => c.startsWith(name + "="))?.split("=")[1];

  async function api(url, options = {}) {
    const headers = { "X-CSRFToken": decodeURIComponent(getCookie("csrftoken") || ""), ...(options.headers || {}) };
    const resp = await fetch(url, { credentials: "same-origin", ...options, headers });
    const body = resp.status === 204 ? null : await resp.json().catch(() => null);
    if (!resp.ok) throw Object.assign(new Error(errorText(body) || `요청 실패 (${resp.status})`), { status: resp.status, body });
    return body;
  }

  // DRF 검증 오류 {"field": ["메시지"]} → 한 줄 메시지
  function errorText(body) {
    if (!body) return "";
    if (typeof body === "string") return body;
    if (body.detail) return body.detail;
    return Object.entries(body).map(([k, v]) => `${k === "non_field_errors" ? "" : k + ": "}${[].concat(v).join(" ")}`).join(" / ");
  }

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (x) => `${Math.round((x ?? 0) * 100)}%`;
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  // ── 이력 표 입력 방식 필터 (분석 화면·관리자 대시보드 공용) ──────────
  let currentFilter = "ALL";
  function applyFilter() {
    $$("#history-body tr[data-input]").forEach((tr) => {
      tr.hidden = currentFilter !== "ALL" && tr.dataset.input !== currentFilter;
    });
    $$(".compare__card").forEach((card) => {
      card.classList.toggle("is-dim", currentFilter !== "ALL" && !card.classList.contains(`compare__card--${currentFilter.toLowerCase()}`));
    });
  }
  $$(".chip[data-filter]").forEach((chip) =>
    chip.addEventListener("click", () => {
      currentFilter = chip.dataset.filter;
      $$(".chip[data-filter]").forEach((c) => c.classList.toggle("is-active", c === chip));
      applyFilter();
    })
  );

  // ── 분석 화면 ─────────────────────────────────────────
  const app = $("#analysis-app");
  if (app) initAnalysisPage(app);

  function initAnalysisPage(app) {
    const form = $("#analysis-form");
    const inputType = form.elements.input_type;
    const startBtn = $("#start-btn");
    const formError = $("#form-error");
    const fileInput = form.elements.video_file;
    const dropzone = $("#dropzone");
    const uploadPreview = $("#upload-preview");
    const maxBytes = Number(app.dataset.maxUploadMb) * 1024 * 1024;
    let pollTimer = null;

    // 탭 전환
    $$(".tab").forEach((tab) =>
      tab.addEventListener("click", () => {
        $$(".tab").forEach((t) => t.classList.toggle("is-active", t === tab));
        $$(".tab-panel").forEach((p) => (p.hidden = p.dataset.panel !== tab.dataset.tab));
        inputType.value = tab.dataset.tab;
        $('#stepper [data-step="CAPTURING"]').hidden = tab.dataset.tab !== "CAMERA";
        hideError();
      })
    );
    $('#stepper [data-step="CAPTURING"]').hidden = true;

    // 업로드: 드래그&드롭 + 미리보기
    ["dragover", "dragenter"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("is-over"); }));
    ["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, () => dropzone.classList.remove("is-over")));
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; onFileChosen(); }
    });
    fileInput.addEventListener("change", onFileChosen);

    function onFileChosen() {
      const f = fileInput.files[0];
      if (!f) return;
      if (f.size > maxBytes) { showError(`파일이 ${app.dataset.maxUploadMb}MB를 넘습니다.`); fileInput.value = ""; return; }
      hideError();
      $("strong", dropzone).textContent = `${f.name} (${(f.size / 1024 / 1024).toFixed(1)}MB)`;
      uploadPreview.src = URL.createObjectURL(f);
      uploadPreview.hidden = false;
    }

    // IP Webcam: 수집 시간 슬라이더 + MJPEG 미리보기 (브라우저가 같은 네트워크의 휴대폰에서 직접 받음)
    const seconds = form.elements.capture_seconds;
    if (seconds) seconds.addEventListener("input", () => ($("#capture-seconds-label").textContent = seconds.value));
    const camBtn = $("#camera-preview-btn");
    if (camBtn) {
      const img = $("#camera-preview");
      camBtn.addEventListener("click", () => {
        if (img.hidden) { img.src = form.elements.camera_url.value; img.hidden = false; camBtn.textContent = "미리보기 끄기"; }
        else { img.removeAttribute("src"); img.hidden = true; camBtn.textContent = "미리보기 켜기"; }
      });
      img.addEventListener("error", () => showError("미리보기를 불러오지 못했습니다. 주소와 휴대폰 앱 상태를 확인하세요. (분석은 서버가 직접 연결합니다)"));
    }

    // 분석 시작
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      hideError();
      const data = new FormData(form);
      if (inputType.value === "UPLOAD") {
        if (!fileInput.files.length) return showError("분석할 영상 파일을 선택하세요.");
        data.delete("camera_url"); data.delete("capture_seconds");
      } else {
        data.delete("video_file");
      }

      setBusy(true);
      renderStage({ stage: "QUEUED", status: "PENDING" }, "요청을 보내는 중…");
      try {
        const job = await api(app.dataset.apiUrl, { method: "POST", body: data });
        $("#job-label").textContent = `#${job.id}`;
        renderResults(null);
        poll(job.id, Date.now());
      } catch (err) {
        setBusy(false);
        renderStage(null, "");
        showError(err.message);
      }
    });

    function poll(jobId, startedAt) {
      clearTimeout(pollTimer);
      api(`${app.dataset.apiUrl}${jobId}/status/`)
        .then((s) => {
          const elapsed = ((Date.now() - startedAt) / 1000).toFixed(0);
          renderStage(s, stageMessage(s, elapsed));
          if (s.is_finished) {
            setBusy(false);
            renderResults(s);
            prependHistory(jobId, s);
          } else {
            pollTimer = setTimeout(() => poll(jobId, startedAt), 1000);
          }
        })
        .catch((err) => { setBusy(false); showError(`상태 조회 실패: ${err.message}`); });
    }

    function stageMessage(s, elapsed) {
      if (s.status === "FAILURE") return `❌ 실패: ${s.error_message}`;
      if (s.status === "SUCCESS") return `✅ ${s.result_message || "완료"} (전체 ${s.total_time_sec ?? "-"}초, AI 추론 ${s.inference_time_sec ?? "-"}초)`;
      if (s.error_message) return `⏳ ${s.error_message} · ${elapsed}초`;
      return { QUEUED: "⏳ 작업 대기열에서 차례를 기다리는 중", CAPTURING: "🎥 카메라에서 영상을 수집하는 중 — 지금 동작하세요!", INFERRING: "🧠 AI가 자세를 추출하고 동작을 예측하는 중" }[s.stage] + ` · ${elapsed}초`;
    }

    function renderStage(s, message) {
      const order = ["QUEUED", "CAPTURING", "INFERRING", "DONE"];
      const idx = s ? order.indexOf(s.stage) : -1;
      $$("#stepper li").forEach((li) => {
        const i = order.indexOf(li.dataset.step);
        li.className = i < idx || (s?.is_finished && i === idx) ? "is-done" : i === idx ? "is-current" : "";
        if (s?.status === "FAILURE" && i === idx) li.className = "is-failed";
      });
      $("#status-line").textContent = message || "분석을 시작하면 진행 상황이 여기에 표시됩니다.";
    }

    function renderResults(s) {
      const tbody = $("#result-table tbody");
      if (!s || !s.predictions?.length) {
        const msg = !s ? "분석 중…" : s.status === "FAILURE" ? "실패한 작업입니다." : (s.result_message || "예측 결과가 없습니다.");
        tbody.innerHTML = `<tr class="empty"><td colspan="4">${esc(msg)}</td></tr>`;
        return;
      }
      tbody.innerHTML = s.predictions.map((p) => `
        <tr>
          <td><strong>${p.track_id}</strong></td>
          <td>${esc(p.action_id)} ${esc(p.action_name_ko)}<br><small class="muted">${esc(p.action_label)}</small></td>
          <td>
            <span class="bar bar--inline"><i style="width:${pct(p.confidence)}"></i></span> ${pct(p.confidence)}
            ${p.is_uncertain ? '<span class="pill pill--warn">판단 불확실</span>' : ""}
            ${p.margin != null ? `<br><small class="muted">2등(${esc(p.runner_up_name_ko || p.runner_up_label)})과 차이 ${(p.margin * 100).toFixed(1)}%p</small>` : ""}
          </td>
          <td><small>유효 Frame ${pct(p.valid_frame_ratio)} · ${p.duration_sec ?? "-"}초<br>${esc(p.window_mode)} · Window ${p.window_count}</small></td>
        </tr>`).join("");
    }

    // 완료된 작업을 비교 · 이력 표 맨 위에 추가 (서버 렌더링과 같은 열 구성)
    function prependHistory(jobId, s) {
      const body = $("#history-body");
      $(".empty", body)?.remove();
      const detail = app.dataset.detailUrl.replace("/0/", `/${jobId}/`);
      const expected = form.elements.expected_action;
      const expectedValue = expected.value;
      const expectedText = expectedValue ? `${expected.options[expected.selectedIndex].text.split(" ")[0]} ${expectedValue}` : "";   // 예: A006 wave
      const isCamera = inputType.value === "CAMERA";

      const preds = s.predictions.length
        ? s.predictions.map((p) => `
            <div class="pred"><span class="muted">T${p.track_id}</span> ${esc(p.action_id)} ${esc(p.action_label)}
              <strong>${pct(p.confidence)}</strong>
              ${p.is_uncertain ? '<span class="pill pill--warn">불확실</span>' : ""}
              ${expectedValue ? (p.action_label === expectedValue ? '<span class="pill pill--success">일치</span>' : '<span class="pill pill--muted">다름</span>') : ""}
            </div>`).join("")
        : `<span class="muted">${esc(s.status === "FAILURE" ? s.error_message : (s.result_message || "-"))}</span>`;
      const time = s.total_time_sec != null
        ? `${Number(s.total_time_sec).toFixed(1)}초${s.inference_time_sec != null ? `<br><small class="muted">AI ${Number(s.inference_time_sec).toFixed(1)}초</small>` : ""}`
        : "-";

      const tr = document.createElement("tr");
      tr.dataset.input = inputType.value;
      tr.innerHTML = `<td><a href="${detail}" title="상세 보기">#${jobId}</a></td><td>${esc(form.elements.title.value || "(방금 요청)")}</td>
        <td class="nowrap">${isCamera ? `IP Webcam <small class="muted">${esc(form.elements.capture_seconds?.value)}초</small>` : "영상 업로드"}</td>
        <td class="nowrap">${expectedText ? esc(expectedText) : '<span class="muted">-</span>'}</td>
        <td>${preds}</td><td>${time}</td>
        <td><span class="pill pill--${s.status.toLowerCase()}">${esc(s.status_display)}</span></td>
        <td>방금</td>`;
      body.prepend(tr);
      applyFilter();
    }

    function setBusy(busy) {
      startBtn.disabled = busy;
      startBtn.textContent = busy ? "분석 중…" : "▶ AI 분석 시작";
    }
    function showError(msg) { formError.textContent = msg; formError.hidden = false; }
    function hideError() { formError.hidden = true; }
  }

  // ── 상세 화면: 제목·메모 수정(PATCH), 삭제(DELETE) ──────────
  const detail = $("#detail-app");
  if (detail) {
    const url = detail.dataset.jobUrl;
    $("#edit-form")?.addEventListener("submit", async (e) => {
      e.preventDefault();
      const f = e.target;
      const msg = $("#edit-msg");
      try {
        await api(url, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: f.elements.title.value, memo: f.elements.memo.value }),
        });
        msg.textContent = "저장했습니다.";
      } catch (err) { msg.textContent = `저장 실패: ${err.message}`; }
    });
    $("#delete-btn")?.addEventListener("click", async () => {
      if (!confirm("이 분석 기록과 사람별 예측, 영상 파일을 삭제할까요? 되돌릴 수 없습니다.")) return;
      try { await api(url, { method: "DELETE" }); location.href = detail.dataset.indexUrl; }
      catch (err) { alert(`삭제 실패: ${err.message}`); }
    });
  }
})();
