/*
 * dashboard.js — Django가 ORM으로 집계해 json_script로 넘긴 데이터로 Chart.js 차트를 그린다.
 *   필수 ① 동작별 예측 건수 (막대, 확정/판단 불확실 누적)   ② 작업 상태별 건수 (도넛)
 *   선택 ③ 입력 방식별 비율 (도넛)                    ④ 입력 방식별 평균 AI 추론 시간 (막대)
 */
(() => {
  "use strict";
  const data = JSON.parse(document.getElementById("chart-data").textContent);
  if (typeof Chart === "undefined") {
    document.querySelectorAll(".chart").forEach((el) => (el.innerHTML = '<p class="muted">Chart.js를 불러오지 못했습니다 (static/vendor/chart.umd.min.js 확인).</p>'));
    return;
  }

  const css = getComputedStyle(document.documentElement);
  const color = (name) => css.getPropertyValue(name).trim();
  Chart.defaults.font.family = css.getPropertyValue("--font").trim();
  Chart.defaults.color = color("--text-2");

  const STATUS_COLORS = { PENDING: color("--gray"), STARTED: color("--blue"), SUCCESS: color("--green"), FAILURE: color("--red") };
  const empty = (arr) => arr.every((v) => !v);

  // ① 동작별 예측 건수 (가로 막대, 누적)
  new Chart(document.getElementById("action-chart"), {
    type: "bar",
    data: {
      labels: data.actions.labels,
      datasets: [
        { label: "확정", data: data.actions.confident, backgroundColor: color("--primary"), borderRadius: 4 },
        { label: "판단 불확실", data: data.actions.uncertain, backgroundColor: color("--primary-soft"), borderRadius: 4 },
      ],
    },
    options: {
      indexAxis: "y",
      maintainAspectRatio: false,
      scales: { x: { stacked: true, beginAtZero: true, ticks: { precision: 0 } }, y: { stacked: true, grid: { display: false } } },
      plugins: { legend: { position: "bottom" } },
    },
  });

  // ② 작업 상태별 건수 (도넛)
  const total = data.status.values.reduce((a, b) => a + b, 0);
  new Chart(document.getElementById("status-chart"), {
    type: "doughnut",
    data: {
      labels: data.status.labels.map((l, i) => `${l} ${data.status.values[i]}`),
      datasets: [{ data: empty(data.status.values) ? [1] : data.status.values, backgroundColor: empty(data.status.values) ? [color("--line")] : data.status.keys.map((k) => STATUS_COLORS[k]), borderWidth: 0 }],
    },
    options: { maintainAspectRatio: false, cutout: "62%", plugins: { legend: { position: "right" }, title: { display: true, text: `총 ${total}건` } } },
  });

  // ③ 입력 방식별 비율
  new Chart(document.getElementById("input-chart"), {
    type: "doughnut",
    data: { labels: data.inputs.labels, datasets: [{ data: empty(data.inputs.values) ? [1, 0] : data.inputs.values, backgroundColor: [color("--primary"), color("--teal")], borderWidth: 0 }] },
    options: { maintainAspectRatio: false, cutout: "62%", plugins: { legend: { position: "right" } } },
  });

  // ④ 입력 방식별 평균 AI 추론 시간
  new Chart(document.getElementById("time-chart"), {
    type: "bar",
    data: { labels: data.inputs.labels, datasets: [{ label: "평균 추론 시간(초)", data: data.inputs.avg_infer, backgroundColor: [color("--primary"), color("--teal")], borderRadius: 4 }] },
    options: { maintainAspectRatio: false, scales: { y: { beginAtZero: true, title: { display: true, text: "초" } }, x: { grid: { display: false } } }, plugins: { legend: { display: false } } },
  });
})();
