const boardEl = document.getElementById('board');
const compareStrip = document.getElementById('compare-strip');
const scenarioSelect = document.getElementById('scenario-select');
const modelSelect = document.getElementById('model-select');
const resultsBody = document.getElementById('results-body');
const fenLabel = document.getElementById('fen-label');
const summaryLabel = document.getElementById('summary-label');
const playToggle = document.getElementById('play-toggle');
const stepBack = document.getElementById('step-back');
const stepForward = document.getElementById('step-forward');
const resetView = document.getElementById('reset-view');

const pieceMap = {
  wK: '♔',
  wQ: '♕',
  wR: '♖',
  wB: '♗',
  wN: '♘',
  wP: '♙',
  bK: '♚',
  bQ: '♛',
  bR: '♜',
  bB: '♝',
  bN: '♞',
  bP: '♟'
};

let playbackFrames = [];
let playbackIndex = 0;
let playbackTimer = null;

function createBoardSquares(fen) {
  boardEl.innerHTML = '';
  const board = fen.split(' ')[0];
  const rows = board.split('/');
  let squareIndex = 0;

  for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
    const row = rows[rowIndex];
    for (let i = 0; i < row.length; i += 1) {
      const ch = row[i];
      if (/[1-8]/.test(ch)) {
        const emptyCount = Number(ch);
        for (let n = 0; n < emptyCount; n += 1) {
          const square = document.createElement('div');
          square.className = `square ${(squareIndex + (Math.floor(squareIndex / 8) % 2)) % 2 === 0 ? 'light' : 'dark'}`;
          boardEl.appendChild(square);
          squareIndex += 1;
        }
      } else {
        const square = document.createElement('div');
        square.className = `square ${(squareIndex + (Math.floor(squareIndex / 8) % 2)) % 2 === 0 ? 'light' : 'dark'}`;
        square.textContent = pieceMap[`${ch === ch.toUpperCase() ? 'w' : 'b'}${pieceName(ch)}`] || '';
        boardEl.appendChild(square);
        squareIndex += 1;
      }
    }
  }
}

function pieceName(ch) {
  if (ch.toLowerCase() === 'k') return 'K';
  if (ch.toLowerCase() === 'q') return 'Q';
  if (ch.toLowerCase() === 'r') return 'R';
  if (ch.toLowerCase() === 'b') return 'B';
  if (ch.toLowerCase() === 'n') return 'N';
  if (ch.toLowerCase() === 'p') return 'P';
  return 'P';
}

function renderPlaybackFrame() {
  if (playbackFrames.length === 0) {
    return;
  }
  const frame = playbackFrames[Math.min(playbackIndex, playbackFrames.length - 1)];
  createBoardSquares(frame.fen);
  fenLabel.textContent = `FEN: ${frame.fen}`;
}

function syncPlaybackFrames() {
  playbackFrames = [
    { label: 'Q-Learning', fen: '' },
    { label: 'DQN', fen: '' },
    { label: 'PPO', fen: '' }
  ];
  const compareRequest = fetch(`/api/compare?scenario=${encodeURIComponent(scenarioSelect.value)}`);
  compareRequest.then((response) => response.json()).then((data) => {
    if (Array.isArray(data.positions)) {
      playbackFrames = data.positions.map((item) => ({ label: item.label, fen: item.fen }));
      playbackIndex = 0;
      renderPlaybackFrame();
      compareStrip.innerHTML = '';
      playbackFrames.forEach((frame) => {
        const chip = document.createElement('div');
        chip.className = 'compare-chip';
        chip.textContent = frame.label;
        chip.addEventListener('click', () => {
          playbackIndex = playbackFrames.indexOf(frame);
          renderPlaybackFrame();
        });
        compareStrip.appendChild(chip);
      });
    }
  });
}

async function loadScenarios() {
  const response = await fetch('/api/scenarios');
  const data = await response.json();
  scenarioSelect.innerHTML = '';
  data.scenarios.forEach((scenario) => {
    const option = document.createElement('option');
    option.value = scenario;
    option.textContent = scenario;
    scenarioSelect.appendChild(option);
  });
}

async function loadModels() {
  const response = await fetch('/api/agents');
  const data = await response.json();
  modelSelect.innerHTML = '';
  data.agents.forEach((agent) => {
    const option = document.createElement('option');
    option.value = agent;
    option.textContent = agent;
    modelSelect.appendChild(option);
  });
}

async function loadFen() {
  const scenario = scenarioSelect.value;
  const response = await fetch(`/api/fen?scenario=${encodeURIComponent(scenario)}`);
  const data = await response.json();
  fenLabel.textContent = `FEN: ${data.fen}`;
  createBoardSquares(data.fen);
  syncPlaybackFrames();
}

function updateSummaryLabel(resultList) {
  if (!Array.isArray(resultList) || resultList.length === 0) {
    summaryLabel.textContent = 'No benchmark data available yet.';
    return;
  }

  const scenarioRows = resultList.filter((entry) => entry.scenario === scenarioSelect.value);
  const current = scenarioRows.find((entry) => entry.algorithm === modelSelect.value) || scenarioRows[0];
  if (!current) {
    summaryLabel.textContent = 'Scenario loaded without matching model data.';
    return;
  }

  const metrics = current.metrics || {};
  summaryLabel.textContent = `mean=${Number(current.mean_reward || metrics.mean_reward || 0).toFixed(3)} | positive=${Number(metrics.positive_rate || 0).toFixed(2)} | zero=${Number(metrics.zero_rate || 0).toFixed(2)} | negative=${Number(metrics.negative_rate || 0).toFixed(2)}`;
}

async function loadResults() {
  const response = await fetch('/api/results');
  const data = await response.json();
  resultsBody.innerHTML = '';
  if (!Array.isArray(data) || data.length === 0) {
    const row = document.createElement('tr');
    row.innerHTML = '<td colspan="5">No results yet.</td>';
    resultsBody.appendChild(row);
    summaryLabel.textContent = 'No benchmark data available yet.';
    return;
  }

  data.forEach((result) => {
    const row = document.createElement('tr');
    const metrics = result.metrics || {};
    const mean = Number(result.mean_reward ?? metrics.mean_reward ?? 0).toFixed(3);
    const positive = Number(metrics.positive_rate ?? 0).toFixed(2);
    const zero = Number(metrics.zero_rate ?? 0).toFixed(2);
    const negative = Number(metrics.negative_rate ?? 0).toFixed(2);
    row.innerHTML = `
      <td>${result.scenario}</td>
      <td>${result.algorithm}</td>
      <td>${mean}</td>
      <td>${positive}</td>
      <td>${zero}/${negative}</td>
    `;
    resultsBody.appendChild(row);
  });

  updateSummaryLabel(data);
}

function togglePlayback() {
  if (playbackFrames.length === 0) {
    return;
  }
  if (playbackTimer) {
    clearInterval(playbackTimer);
    playbackTimer = null;
    playToggle.textContent = 'Play';
    return;
  }
  playToggle.textContent = 'Pause';
  playbackTimer = setInterval(() => {
    playbackIndex = (playbackIndex + 1) % playbackFrames.length;
    renderPlaybackFrame();
  }, 850);
}

function stepPlayback(direction) {
  if (playbackFrames.length === 0) {
    return;
  }
  if (playbackTimer) {
    clearInterval(playbackTimer);
    playbackTimer = null;
    playToggle.textContent = 'Play';
  }
  playbackIndex = (playbackIndex + direction + playbackFrames.length) % playbackFrames.length;
  renderPlaybackFrame();
}

scenarioSelect.addEventListener('change', () => {
  loadFen();
  loadResults();
});
modelSelect.addEventListener('change', loadResults);
playToggle.addEventListener('click', togglePlayback);
stepBack.addEventListener('click', () => stepPlayback(-1));
stepForward.addEventListener('click', () => stepPlayback(1));
resetView.addEventListener('click', () => {
  playbackIndex = 0;
  renderPlaybackFrame();
});

(async () => {
  await loadScenarios();
  await loadModels();
  await loadFen();
  await loadResults();
})();
