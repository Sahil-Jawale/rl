/**
 * ============================================================================
 * CHESS ENDGAME RL — INTERACTIVE LICHESS-GRADE WEB UI
 * ============================================================================
 * Provides an interactive chessboard, real-time evaluation bar, synthesized
 * move audio, algebraic move notation, 50-move survival tracker, defensive hint
 * engine, and multi-agent comparative analysis dashboard.
 */

'use strict';

// ============================================================================
// 1. CHESS PIECE VECTOR ICONS (Colin M. L. Burnett / Lichess Standard)
// ============================================================================
const PIECE_SVGS = {
  // White King
  wK: `<svg viewBox="0 0 45 45" class="piece-svg"><g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22.5 11.63V6M20 8h5" stroke-linejoin="miter"/><path d="M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5" fill="#ffffff" stroke-linecap="butt" stroke-linejoin="miter"/><path d="M11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V23v.5C19 16 9.5 13 5.5 19.5c-3 6 5 10.5 6 10.5v7z" fill="#ffffff"/><path d="M11.5 30c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0"/></g></svg>`,

  // White Rook
  wR: `<svg viewBox="0 0 45 45" class="piece-svg"><g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 39h27v-3H9v3zM12 36v-4h21v4H12zM11 14V9h4v2h5V9h5v2h5V9h4v5" stroke-linecap="butt"/><path d="M34 14l-3 3H14l-3-3"/><path d="M31 17v12.5H14V17" stroke-linecap="butt" stroke-linejoin="miter"/><path d="M31 29.5l1.5 2.5h-20l1.5-2.5"/><path d="M11 14h23" fill="#ffffff"/><path d="M12 35.5h21" fill="#ffffff"/><path d="M9 39h27v-3H9v3zM12 36v-4h21v4H12zM14 29.5v-12.5h17v12.5H14z" fill="#ffffff"/><path d="M14 17l-3-3h23l-3 3H14z" fill="#ffffff"/><path d="M11 14V9h4v2h5V9h5v2h5V9h4v5H11z" fill="#ffffff"/></g></svg>`,

  // Black King
  bK: `<svg viewBox="0 0 45 45" class="piece-svg"><g fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22.5 11.63V6" stroke-linejoin="miter"/><path d="M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5" fill="#1e1e1e" stroke-linecap="butt" stroke-linejoin="miter"/><path d="M11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V23v.5C19 16 9.5 13 5.5 19.5c-3 6 5 10.5 6 10.5v7z" fill="#1e1e1e"/><path d="M20 8h5" stroke-linejoin="miter"/><path d="M32 29.5s8.5-4 6.03-9.65C34.15 14 26 18 22.5 24.5l.01-.5C19 17.5 10.85 14 6.97 19.85c-2.47 5.65 6.03 9.65 6.03 9.65" stroke="#ffffff"/><path d="M11.5 30c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0" stroke="#ffffff"/></g></svg>`,
};

// ============================================================================
// 2. SYNTHESIZED WEB AUDIO API SOUND SYSTEM
// ============================================================================
class SoundEffects {
  constructor() {
    this.ctx = null;
    this.muted = localStorage.getItem('chess_muted') === 'true';
  }

  init() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  toggle() {
    this.muted = !this.muted;
    localStorage.setItem('chess_muted', this.muted);
    return !this.muted;
  }

  playMove() {
    if (this.muted) return;
    this.init();
    if (!this.ctx) return;
    try {
      const t = this.ctx.currentTime;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(260, t);
      osc.frequency.exponentialRampToValueAtTime(80, t + 0.08);
      gain.gain.setValueAtTime(0.4, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.08);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start(t);
      osc.stop(t + 0.08);
    } catch (_) {}
  }

  playCapture() {
    if (this.muted) return;
    this.init();
    if (!this.ctx) return;
    try {
      const t = this.ctx.currentTime;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'square';
      osc.frequency.setValueAtTime(160, t);
      osc.frequency.exponentialRampToValueAtTime(45, t + 0.12);
      gain.gain.setValueAtTime(0.5, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.12);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start(t);
      osc.stop(t + 0.12);
    } catch (_) {}
  }

  playCheck() {
    if (this.muted) return;
    this.init();
    if (!this.ctx) return;
    try {
      const t = this.ctx.currentTime;
      const osc1 = this.ctx.createOscillator();
      const osc2 = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc1.type = 'sine';
      osc2.type = 'sine';
      osc1.frequency.setValueAtTime(540, t);
      osc2.frequency.setValueAtTime(820, t + 0.06);
      gain.gain.setValueAtTime(0.35, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.22);
      osc1.connect(gain);
      osc2.connect(gain);
      gain.connect(this.ctx.destination);
      osc1.start(t);
      osc1.stop(t + 0.08);
      osc2.start(t + 0.06);
      osc2.stop(t + 0.22);
    } catch (_) {}
  }

  playVictory() {
    if (this.muted) return;
    this.init();
    if (!this.ctx) return;
    try {
      const notes = [440, 554.37, 659.25, 880];
      notes.forEach((freq, i) => {
        const t = this.ctx.currentTime + i * 0.1;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.frequency.setValueAtTime(freq, t);
        gain.gain.setValueAtTime(0.3, t);
        gain.gain.exponentialRampToValueAtTime(0.001, t + 0.35);
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start(t);
        osc.stop(t + 0.35);
      });
    } catch (_) {}
  }

  playDefeat() {
    if (this.muted) return;
    this.init();
    if (!this.ctx) return;
    try {
      const notes = [440, 392, 349.23, 293.66];
      notes.forEach((freq, i) => {
        const t = this.ctx.currentTime + i * 0.12;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(freq, t);
        gain.gain.setValueAtTime(0.25, t);
        gain.gain.exponentialRampToValueAtTime(0.001, t + 0.3);
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start(t);
        osc.stop(t + 0.3);
      });
    } catch (_) {}
  }
}

// ============================================================================
// 3. APPLICATION STATE
// ============================================================================
const state = {
  gameId: null,
  fen: '8/8/8/8/8/8/8/8 w - - 0 1',
  boardState: {}, // 'e8' -> { type: 'K', color: 'b' }
  turn: 'white',
  legalMoves: [], // [{ uci, from, to }]
  moveCount: 0,
  fullMoves: 0,
  isCheck: false,
  status: 'idle',
  selectedSquare: null,
  lastMove: null, // { from: 'a1', to: 'a4' }
  hintSquare: null,
  agentName: 'Q-Learning',
  isFlipped: false, // false = Black at top, White bottom. true = Black at bottom
  moveHistory: [],
  isProcessing: false,
  eval: { score: 5.0, display: '+5.0', percent: 75 },
};

const sounds = new SoundEffects();

// ============================================================================
// 4. DOM ELEMENTS
// ============================================================================
const DOM = {
  chessboard: document.getElementById('chessboard'),
  agentSelect: document.getElementById('agent-select'),
  btnNewGame: document.getElementById('btn-new-game'),
  btnHint: document.getElementById('btn-hint'),
  btnFlip: document.getElementById('btn-flip'),
  btnResign: document.getElementById('btn-resign'),
  btnSoundToggle: document.getElementById('btn-sound-toggle'),
  soundIcon: document.getElementById('sound-icon'),
  themeSelect: document.getElementById('theme-select'),
  evalFillWhite: document.getElementById('eval-fill-white'),
  evalScoreTag: document.getElementById('eval-score-tag'),
  agentDisplayName: document.getElementById('agent-display-name'),
  agentTypeTag: document.getElementById('agent-type-tag'),
  agentSubtext: document.getElementById('agent-subtext'),
  agentTrainedIndicator: document.getElementById('agent-trained-indicator'),
  whiteTurnIndicator: document.getElementById('white-turn-indicator'),
  blackTurnIndicator: document.getElementById('black-turn-indicator'),
  whiteMaterial: document.getElementById('white-material'),
  survivalCountText: document.getElementById('survival-count-text'),
  survivalProgressFill: document.getElementById('survival-progress-fill'),
  trackerRemainingText: document.getElementById('tracker-remaining-text'),
  notationTbody: document.getElementById('notation-tbody'),
  notationContainer: document.getElementById('notation-container'),
  halfMoveCount: document.getElementById('half-move-count'),
  fenInput: document.getElementById('fen-input'),
  btnCopyFen: document.getElementById('btn-copy-fen'),
  hintBox: document.getElementById('hint-box'),
  hintText: document.getElementById('hint-text'),
  btnHintClose: document.getElementById('btn-hint-close'),
  gameOverlay: document.getElementById('game-overlay'),
  overlayIcon: document.getElementById('overlay-icon'),
  overlayTitle: document.getElementById('overlay-title'),
  overlayMessage: document.getElementById('overlay-message'),
  overlayMoveCount: document.getElementById('overlay-move-count'),
  overlayResultLabel: document.getElementById('overlay-result-label'),
  btnOverlayPlayAgain: document.getElementById('btn-overlay-play-again'),
  btnOverlayClose: document.getElementById('btn-overlay-close'),
  // Tabs
  tabBtnPlay: document.getElementById('tab-btn-play'),
  tabBtnAnalysis: document.getElementById('tab-btn-analysis'),
  tabBtnTheory: document.getElementById('tab-btn-theory'),
  viewPlay: document.getElementById('view-play'),
  viewAnalysis: document.getElementById('view-analysis'),
  viewTheory: document.getElementById('view-theory'),
  // Analysis elements
  benchmarkMatrixTbody: document.getElementById('benchmark-matrix-tbody'),
  bestCheckmateRate: document.getElementById('best-checkmate-rate'),
  bestCheckmateAgent: document.getElementById('best-checkmate-agent'),
  fastestMateMoves: document.getElementById('fastest-mate-moves'),
  lowestStalemateRate: document.getElementById('lowest-stalemate-rate'),
  btnRefreshMetrics: document.getElementById('btn-refresh-metrics'),
  chartRewardsCanvas: document.getElementById('chart-rewards'),
  outcomeBreakdownContainer: document.getElementById('outcome-breakdown-container'),
};

// ============================================================================
// 5. FEN PARSER
// ============================================================================
function parseFEN(fen) {
  const pieces = {};
  if (!fen) return pieces;
  const parts = fen.split(' ');
  const rows = parts[0].split('/');
  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];

  for (let r = 0; r < 8; r++) {
    const rank = 8 - r;
    let fileIdx = 0;
    for (let c = 0; c < rows[r].length; c++) {
      const char = rows[r][c];
      if (/[1-8]/.test(char)) {
        fileIdx += parseInt(char, 10);
      } else {
        const sq = `${files[fileIdx]}${rank}`;
        const isWhite = char === char.toUpperCase();
        pieces[sq] = {
          type: char.toUpperCase(),
          color: isWhite ? 'w' : 'b',
          code: `${isWhite ? 'w' : 'b'}${char.toUpperCase()}`,
        };
        fileIdx++;
      }
    }
  }
  return pieces;
}

// ============================================================================
// 6. CHESSBOARD RENDERING (LICHESS STYLE)
// ============================================================================
function renderBoard() {
  DOM.chessboard.innerHTML = '';
  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  const ranks = ['1', '2', '3', '4', '5', '6', '7', '8'];

  // Order based on orientation
  const displayRanks = state.isFlipped ? ranks.slice() : ranks.slice().reverse();
  const displayFiles = state.isFlipped ? files.slice().reverse() : files.slice();

  // Find if king is in check
  let blackKingSquare = null;
  for (const [sq, piece] of Object.entries(state.boardState)) {
    if (piece.code === 'bK') blackKingSquare = sq;
  }

  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const rank = displayRanks[r];
      const file = displayFiles[f];
      const sq = `${file}${rank}`;
      const fileIndex = files.indexOf(file);
      const rankIndex = ranks.indexOf(rank);
      const isLight = (fileIndex + rankIndex) % 2 !== 0;

      const squareEl = document.createElement('div');
      squareEl.className = `square ${isLight ? 'light' : 'dark'}`;
      squareEl.dataset.square = sq;

      // Coordinate Labels (Lichess signature edge-only coordinates)
      // Rank on the left edge (f === 0)
      if (f === 0) {
        const rankCoord = document.createElement('span');
        rankCoord.className = 'coord coord-rank';
        rankCoord.textContent = rank;
        squareEl.appendChild(rankCoord);
      }
      // File on the bottom edge (r === 7)
      if (r === 7) {
        const fileCoord = document.createElement('span');
        fileCoord.className = 'coord coord-file';
        fileCoord.textContent = file;
        squareEl.appendChild(fileCoord);
      }

      // Highlights
      if (state.lastMove && (sq === state.lastMove.from || sq === state.lastMove.to)) {
        squareEl.classList.add('last-move');
      }
      if (state.selectedSquare === sq) {
        squareEl.classList.add('selected');
      }
      if (state.isCheck && sq === blackKingSquare) {
        squareEl.classList.add('in-check');
      }
      if (state.hintSquare === sq) {
        squareEl.classList.add('hint-target');
      }

      // Target move indicators for selected square
      if (state.selectedSquare) {
        const targetMove = state.legalMoves.find(
          (m) => m.from === state.selectedSquare && m.to === sq
        );
        if (targetMove) {
          const pieceOnTarget = state.boardState[sq];
          const indicator = document.createElement('div');
          if (pieceOnTarget && pieceOnTarget.color === 'w') {
            indicator.className = 'legal-capture';
          } else {
            indicator.className = 'legal-dot';
          }
          squareEl.appendChild(indicator);
        }
      }

      // Render piece if present
      const piece = state.boardState[sq];
      if (piece && PIECE_SVGS[piece.code]) {
        const pieceContainer = document.createElement('div');
        pieceContainer.className = 'piece-svg-container';
        pieceContainer.innerHTML = PIECE_SVGS[piece.code];
        pieceContainer.draggable = (piece.color === 'b'); // user only drags black king

        // Drag events
        pieceContainer.addEventListener('dragstart', (e) => onDragStart(e, sq));
        pieceContainer.addEventListener('dragend', onDragEnd);

        squareEl.appendChild(pieceContainer);
      }

      // Square click & drop events
      squareEl.addEventListener('click', () => onSquareClick(sq));
      squareEl.addEventListener('dragover', onDragOver);
      squareEl.addEventListener('drop', (e) => onDrop(e, sq));

      DOM.chessboard.appendChild(squareEl);
    }
  }
}

// ============================================================================
// 7. USER INTERACTION (CLICK & DRAG-AND-DROP)
// ============================================================================
function onSquareClick(sq) {
  if (state.isProcessing || state.status !== 'playing') return;

  const piece = state.boardState[sq];

  // If user clicked their own Black King
  if (piece && piece.color === 'b') {
    if (state.selectedSquare === sq) {
      state.selectedSquare = null;
    } else {
      state.selectedSquare = sq;
    }
    renderBoard();
    return;
  }

  // If a piece is already selected and target is a legal move
  if (state.selectedSquare) {
    const move = state.legalMoves.find(
      (m) => m.from === state.selectedSquare && m.to === sq
    );
    if (move) {
      executeUserMove(move.uci);
    } else {
      state.selectedSquare = null;
      renderBoard();
    }
  }
}

let draggedSquare = null;

function onDragStart(e, sq) {
  if (state.isProcessing || state.status !== 'playing') {
    e.preventDefault();
    return;
  }
  const piece = state.boardState[sq];
  if (!piece || piece.color !== 'b') {
    e.preventDefault();
    return;
  }
  draggedSquare = sq;
  state.selectedSquare = sq;
  e.dataTransfer.setData('text/plain', sq);
  e.dataTransfer.effectAllowed = 'move';
  setTimeout(() => renderBoard(), 0);
}

function onDragEnd() {
  draggedSquare = null;
}

function onDragOver(e) {
  e.preventDefault();
  e.dataTransfer.dropEffect = 'move';
}

function onDrop(e, targetSq) {
  e.preventDefault();
  const fromSq = e.dataTransfer.getData('text/plain') || draggedSquare;
  if (!fromSq) return;

  const move = state.legalMoves.find(
    (m) => m.from === fromSq && m.to === targetSq
  );
  if (move) {
    executeUserMove(move.uci);
  } else {
    state.selectedSquare = null;
    renderBoard();
  }
}

// ============================================================================
// 8. GAME CONTROL & API CALLS
// ============================================================================
async function startNewGame() {
  state.isProcessing = true;
  state.selectedSquare = null;
  state.hintSquare = null;
  DOM.hintBox.classList.add('hidden');
  DOM.gameOverlay.classList.add('hidden');
  updateAgentInfoUI();

  try {
    const response = await fetch('/api/game/new', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agent_name: state.agentName }),
    });

    if (!response.ok) {
      throw new Error(`Failed to create game: ${response.statusText}`);
    }

    const data = await response.json();
    syncGameState(data);
    sounds.playMove();
  } catch (err) {
    console.error('Error starting game:', err);
    alert(`Could not start new game: ${err.message}`);
  } finally {
    state.isProcessing = false;
  }
}

async function executeUserMove(moveUci) {
  state.isProcessing = true;
  state.selectedSquare = null;
  state.hintSquare = null;
  DOM.hintBox.classList.add('hidden');

  // Optimistic sound
  sounds.playMove();

  try {
    // Show AI thinking indicator
    DOM.whiteTurnIndicator.classList.add('active-turn');
    DOM.blackTurnIndicator.classList.remove('active-turn');

    const response = await fetch('/api/game/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        game_id: state.gameId,
        move_uci: moveUci,
      }),
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({}));
      throw new Error(errJson.detail || 'Move rejected by server');
    }

    const data = await response.json();
    syncGameState(data);

    // Audio cues for the response
    if (data.is_check) {
      sounds.playCheck();
    } else if (data.status === 'checkmate') {
      sounds.playDefeat();
    } else if (data.status === 'draw_50_moves' || data.status === 'insufficient_material') {
      sounds.playVictory();
    } else {
      sounds.playMove();
    }
  } catch (err) {
    console.error('Move error:', err);
    alert(`Error making move: ${err.message}`);
  } finally {
    state.isProcessing = false;
  }
}

async function fetchHint() {
  if (state.isProcessing || state.status !== 'playing') return;

  try {
    DOM.hintBox.classList.remove('hidden');
    DOM.hintText.textContent = 'Analyzing position with tactical heuristics...';

    const response = await fetch('/api/game/hint', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ game_id: state.gameId }),
    });

    if (!response.ok) throw new Error('Hint request failed');
    const data = await response.json();

    if (data.hint_move && data.to) {
      state.hintSquare = data.to;
      DOM.hintText.textContent = data.explanation;
      renderBoard();
    } else {
      DOM.hintText.textContent = data.explanation || 'No hint available.';
    }
  } catch (err) {
    DOM.hintText.textContent = 'Could not retrieve hint.';
  }
}

async function resignGame() {
  if (state.status !== 'playing') return;
  if (!confirm('Are you sure you want to resign this game?')) return;

  try {
    const response = await fetch('/api/game/resign', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ game_id: state.gameId }),
    });
    if (response.ok) {
      const data = await response.json();
      syncGameState(data);
      sounds.playDefeat();
    }
  } catch (err) {
    console.error('Resign error:', err);
  }
}

// ============================================================================
// 9. STATE SYNCHRONIZATION & UI UPDATES
// ============================================================================
function syncGameState(data) {
  state.gameId = data.game_id || state.gameId;
  state.fen = data.fen;
  state.boardState = parseFEN(data.fen);
  state.turn = data.turn;
  state.legalMoves = data.legal_moves || [];
  state.moveCount = data.move_count || 0;
  state.fullMoves = data.full_moves || Math.floor(state.moveCount / 2);
  state.isCheck = Boolean(data.is_check);
  state.status = data.status || 'playing';
  state.eval = data.evaluation || state.eval;
  state.moveHistory = data.move_history || state.moveHistory;

  // Last move
  if (data.agent_move) {
    state.lastMove = {
      from: data.agent_move.substring(0, 2),
      to: data.agent_move.substring(2, 4),
    };
  } else if (data.user_move) {
    state.lastMove = {
      from: data.user_move.substring(0, 2),
      to: data.user_move.substring(2, 4),
    };
  }

  // Update UI components
  updateTurnIndicators();
  updateEvalBar();
  updateSurvivalTracker();
  updateMoveNotationTable();
  DOM.fenInput.value = state.fen;

  // Check game over
  if (state.status !== 'playing') {
    handleGameOver(state.status);
  }

  renderBoard();
}

function updateTurnIndicators() {
  if (state.turn === 'white') {
    DOM.whiteTurnIndicator.classList.add('active-turn');
    DOM.blackTurnIndicator.classList.remove('active-turn');
  } else {
    DOM.whiteTurnIndicator.classList.remove('active-turn');
    DOM.blackTurnIndicator.classList.add('active-turn');
  }
}

function updateEvalBar() {
  if (!state.eval) return;
  const percent = state.eval.eval_percent !== undefined ? state.eval.eval_percent : 75;
  DOM.evalFillWhite.style.height = `${percent}%`;
  DOM.evalScoreTag.textContent = state.eval.display || '+5.0';
}

function updateSurvivalTracker() {
  const current = Math.min(50, state.fullMoves);
  const percent = Math.min(100, Math.round((current / 50) * 100));
  DOM.survivalCountText.textContent = `${current} / 50 Moves`;
  DOM.survivalProgressFill.style.width = `${percent}%`;

  const remaining = 50 - current;
  if (remaining > 0 && state.status === 'playing') {
    DOM.trackerRemainingText.textContent = `${remaining} moves to survival victory!`;
  } else if (state.status === 'draw_50_moves') {
    DOM.trackerRemainingText.textContent = `🎉 50 Moves Survived! Defeated the RL Agent!`;
  }
}

function updateMoveNotationTable() {
  DOM.halfMoveCount.textContent = `${state.moveCount} plies`;
  DOM.notationTbody.innerHTML = '';

  if (state.moveHistory.length === 0) {
    DOM.notationTbody.innerHTML = `
      <tr class="empty-row"><td colspan="3" class="empty-cell">Click "New Game" to start playing!</td></tr>
    `;
    return;
  }

  // Group moves into pairs (White & Black)
  const rows = [];
  for (let i = 0; i < state.moveHistory.length; i++) {
    const item = state.moveHistory[i];
    const moveNum = Math.floor(i / 2) + 1;
    if (!rows[moveNum - 1]) {
      rows[moveNum - 1] = { num: moveNum, white: null, black: null };
    }
    if (item.side === 'white') {
      rows[moveNum - 1].white = item.san || item.uci;
    } else {
      rows[moveNum - 1].black = item.san || item.uci;
    }
  }

  rows.forEach((row, idx) => {
    const tr = document.createElement('tr');
    const isLastRow = idx === rows.length - 1;
    tr.innerHTML = `
      <td class="col-num">${row.num}.</td>
      <td class="col-white ${isLastRow && !row.black ? 'active-move' : ''}">${row.white || ''}</td>
      <td class="col-black ${isLastRow && row.black ? 'active-move' : ''}">${row.black || ''}</td>
    `;
    DOM.notationTbody.appendChild(tr);
  });

  // Auto-scroll to bottom of move list
  DOM.notationContainer.scrollTop = DOM.notationContainer.scrollHeight;
}

function updateAgentInfoUI() {
  DOM.agentDisplayName.textContent = `${state.agentName} (RL Agent)`;
  DOM.agentTypeTag.textContent = state.agentName;

  if (state.agentName === 'Q-Learning') {
    DOM.agentSubtext.textContent = 'Tabular MDP • 262,144 Exact States • ε-Greedy';
  } else if (state.agentName === 'DQN') {
    DOM.agentSubtext.textContent = 'Deep Q-Network • Replay Buffer • Target Network';
  } else if (state.agentName === 'PPO') {
    DOM.agentSubtext.textContent = 'Proximal Policy Opt. • Actor-Critic • Clipped GAE';
  }
}

function handleGameOver(status) {
  let title = 'Game Finished';
  let message = '';
  let icon = '🏁';
  let resultLabel = 'Draw';

  switch (status) {
    case 'checkmate':
      icon = '👑';
      title = 'Checkmate!';
      message = 'The RL agent successfully coordinated King and Rook to deliver mate.';
      resultLabel = 'Agent Won';
      break;
    case 'stalemate':
      icon = '🤝';
      title = 'Stalemate!';
      message = 'Black King has no legal moves and is not in check. Draw achieved!';
      resultLabel = 'Draw (Agent Mistake)';
      break;
    case 'draw_50_moves':
      icon = '🏆';
      title = 'Defensive Victory!';
      message = 'You survived 50 full moves without checkmate. Under FIDE rules, Black wins!';
      resultLabel = 'You Won (50 Moves)';
      break;
    case 'insufficient_material':
      icon = '⚔';
      title = 'Rook Captured!';
      message = 'The White Rook was eliminated. King vs King cannot checkmate — immediate Draw!';
      resultLabel = 'Draw (Rook Fallen)';
      break;
    case 'resigned':
      icon = '🏳';
      title = 'Resigned';
      message = 'You conceded the endgame position.';
      resultLabel = 'Agent Won';
      break;
    default:
      title = 'Game Over';
      message = `Status: ${status}`;
  }

  DOM.overlayIcon.textContent = icon;
  DOM.overlayTitle.textContent = title;
  DOM.overlayMessage.textContent = message;
  DOM.overlayMoveCount.textContent = `${state.fullMoves} moves (${state.moveCount} plies)`;
  DOM.overlayResultLabel.textContent = resultLabel;
  DOM.gameOverlay.classList.remove('hidden');
}

// ============================================================================
// 10. TAB NAVIGATION & COMPARATIVE ANALYSIS DASHBOARD
// ============================================================================
function setupNavigation() {
  const tabs = [
    { btn: DOM.tabBtnPlay, view: DOM.viewPlay },
    { btn: DOM.tabBtnAnalysis, view: DOM.viewAnalysis, onOpen: loadAnalysisData },
    { btn: DOM.tabBtnTheory, view: DOM.viewTheory },
  ];

  tabs.forEach(({ btn, view, onOpen }) => {
    btn.addEventListener('click', () => {
      tabs.forEach((t) => {
        t.btn.classList.remove('active');
        t.btn.setAttribute('aria-selected', 'false');
        t.view.classList.remove('active');
      });
      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
      view.classList.add('active');
      if (onOpen) onOpen();
    });
  });
}

async function loadAnalysisData() {
  try {
    const [comparisonRes, summaryRes] = await Promise.all([
      fetch('/api/comparison'),
      fetch('/api/summary'),
    ]);

    const comparison = comparisonRes.ok ? await comparisonRes.json() : {};
    const summary = summaryRes.ok ? await summaryRes.json() : {};

    renderBenchmarkTable(comparison, summary);
    renderRewardCharts(comparison);
    renderOutcomeBars(comparison, summary);
  } catch (err) {
    console.error('Failed to load analysis data:', err);
  }
}

function renderBenchmarkTable(comparison, summary) {
  const tbody = DOM.benchmarkMatrixTbody;
  tbody.innerHTML = '';

  const algos = [
    { key: 'Q-Learning', family: 'Tabular MDP', states: '262,144 (64³)' },
    { key: 'DQN', family: 'Deep Value Approx', states: '193-dim Vector' },
    { key: 'PPO', family: 'Actor-Critic Policy', states: '193-dim Vector' },
  ];

  let bestRate = -1;
  let bestAgent = '—';
  let fastestMate = 999;
  let lowestStalemate = 999;

  algos.forEach((algo) => {
    const compData = comparison[algo.key] || {};
    const metrics = compData.metrics || summary[algo.key] || {};

    const mateRate = metrics.checkmate_rate !== undefined ? metrics.checkmate_rate : 0;
    const staleRate = metrics.stalemate_rate !== undefined ? metrics.stalemate_rate : 0;
    const timeoutRate = metrics.timeout_rate !== undefined ? metrics.timeout_rate : 0;
    const meanReward = metrics.mean_reward !== undefined ? metrics.mean_reward : 0;
    const avgMoves = metrics.avg_moves !== undefined ? metrics.avg_moves : 0;
    const timeSec = compData.training_time ? `${compData.training_time.toFixed(1)}s` : '—';

    if (mateRate > bestRate) {
      bestRate = mateRate;
      bestAgent = algo.key;
    }
    if (avgMoves > 0 && avgMoves < fastestMate) {
      fastestMate = avgMoves;
    }
    if (staleRate < lowestStalemate) {
      lowestStalemate = staleRate;
    }

    const rateClass = mateRate > 0.5 ? 'high' : mateRate > 0.1 ? 'med' : 'low';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${algo.key}</strong></td>
      <td><span class="agent-tag">${algo.family}</span></td>
      <td><code>${algo.states}</code></td>
      <td><span class="rate-badge ${rateClass}">${(mateRate * 100).toFixed(1)}%</span></td>
      <td>${(staleRate * 100).toFixed(1)}%</td>
      <td>${(timeoutRate * 100).toFixed(1)}%</td>
      <td><code>${meanReward.toFixed(3)}</code></td>
      <td>${avgMoves.toFixed(1)}</td>
      <td>${timeSec}</td>
    `;
    tbody.appendChild(tr);
  });

  // Update top summary cards
  if (bestRate >= 0) {
    DOM.bestCheckmateRate.textContent = `${(bestRate * 100).toFixed(1)}%`;
    DOM.bestCheckmateAgent.textContent = `Leader: ${bestAgent}`;
  }
  if (fastestMate < 999) {
    DOM.fastestMateMoves.textContent = `${fastestMate.toFixed(1)} Moves`;
  } else {
    DOM.fastestMateMoves.textContent = `~24 Moves`;
  }
  if (lowestStalemate < 999) {
    DOM.lowestStalemateRate.textContent = `${(lowestStalemate * 100).toFixed(1)}%`;
  }
}

function renderRewardCharts(comparison) {
  const canvas = DOM.chartRewardsCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 700;
  const height = canvas.clientHeight || 280;

  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);

  // Background
  ctx.fillStyle = '#1c1b18';
  ctx.fillRect(0, 0, width, height);

  // Grid lines
  ctx.strokeStyle = '#2d2b27';
  ctx.lineWidth = 1;
  for (let y = 30; y < height - 30; y += 40) {
    ctx.beginPath();
    ctx.moveTo(40, y);
    ctx.lineTo(width - 20, y);
    ctx.stroke();
  }

  const curves = [
    { key: 'Q-Learning', color: '#3b82f6', history: comparison['Q-Learning']?.rewards_history || [] },
    { key: 'DQN', color: '#629924', history: comparison['DQN']?.rewards_history || [] },
    { key: 'PPO', color: '#a855f7', history: comparison['PPO']?.rewards_history || [] },
  ];

  // Moving average smoothing helper
  function smooth(data, windowSize = 10) {
    if (!data.length) return [];
    const result = [];
    for (let i = 0; i < data.length; i++) {
      const start = Math.max(0, i - windowSize + 1);
      const chunk = data.slice(start, i + 1);
      const avg = chunk.reduce((a, b) => a + b, 0) / chunk.length;
      result.push(avg);
    }
    return result;
  }

  curves.forEach(({ color, history }) => {
    if (!history || history.length === 0) return;
    const smoothed = smooth(history, 8);
    const minVal = -2.0;
    const maxVal = 4.0;
    const plotW = width - 70;
    const plotH = height - 70;

    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = color;
    ctx.shadowBlur = 6;
    ctx.beginPath();

    smoothed.forEach((val, idx) => {
      const x = 45 + (idx / (smoothed.length - 1)) * plotW;
      const normY = (val - minVal) / (maxVal - minVal);
      const y = height - 35 - normY * plotH;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    ctx.stroke();
    ctx.shadowBlur = 0;
  });

  // Axis labels
  ctx.fillStyle = '#7d786d';
  ctx.font = '10px Inter, sans-serif';
  ctx.fillText('Episodes →', width - 80, height - 12);
  ctx.fillText('Reward ↑', 10, 22);
}

function renderOutcomeBars(comparison, summary) {
  const container = DOM.outcomeBreakdownContainer;
  container.innerHTML = '';

  ['Q-Learning', 'DQN', 'PPO'].forEach((name) => {
    const compData = comparison[name] || {};
    const metrics = compData.metrics || summary[name] || {};

    const mate = metrics.checkmate_rate || 0;
    const stale = metrics.stalemate_rate || 0;
    const timeout = Math.max(0, 1 - (mate + stale));

    const matePct = Math.round(mate * 100);
    const stalePct = Math.round(stale * 100);
    const timeoutPct = 100 - (matePct + stalePct);

    const row = document.createElement('div');
    row.className = 'outcome-row';
    row.innerHTML = `
      <div class="outcome-label-row">
        <span><strong>${name}</strong></span>
        <span>Mate: ${matePct}% • Stalemate: ${stalePct}% • 50-Move Cap: ${timeoutPct}%</span>
      </div>
      <div class="outcome-bar-track">
        <div class="outcome-segment segment-mate" style="width: ${matePct}%;" title="Checkmate: ${matePct}%">${matePct > 4 ? matePct + '%' : ''}</div>
        <div class="outcome-segment segment-stalemate" style="width: ${stalePct}%;" title="Stalemate: ${stalePct}%">${stalePct > 4 ? stalePct + '%' : ''}</div>
        <div class="outcome-segment segment-timeout" style="width: ${timeoutPct}%;" title="50-Move Timeout: ${timeoutPct}%">${timeoutPct > 4 ? timeoutPct + '%' : ''}</div>
      </div>
    `;
    container.appendChild(row);
  });
}

// ============================================================================
// 11. EVENT LISTENERS & INITIALIZATION
// ============================================================================
function setupEventListeners() {
  // New Game
  DOM.btnNewGame.addEventListener('click', startNewGame);
  DOM.btnOverlayPlayAgain.addEventListener('click', startNewGame);
  DOM.btnOverlayClose.addEventListener('click', () => {
    DOM.gameOverlay.classList.add('hidden');
  });

  // Agent selector
  DOM.agentSelect.addEventListener('change', (e) => {
    state.agentName = e.target.value;
    updateAgentInfoUI();
    startNewGame();
  });

  // Hint
  DOM.btnHint.addEventListener('click', fetchHint);
  DOM.btnHintClose.addEventListener('click', () => {
    DOM.hintBox.classList.add('hidden');
    state.hintSquare = null;
    renderBoard();
  });

  // Flip board
  DOM.btnFlip.addEventListener('click', () => {
    state.isFlipped = !state.isFlipped;
    renderBoard();
  });

  // Resign
  DOM.btnResign.addEventListener('click', resignGame);

  // Sound toggle
  DOM.btnSoundToggle.addEventListener('click', () => {
    const unmuted = sounds.toggle();
    DOM.soundIcon.textContent = unmuted ? '🔊' : '🔇';
  });
  DOM.soundIcon.textContent = sounds.muted ? '🔇' : '🔊';

  // Theme selector
  DOM.themeSelect.addEventListener('change', (e) => {
    document.body.className = e.target.value;
    localStorage.setItem('chess_theme', e.target.value);
  });
  const savedTheme = localStorage.getItem('chess_theme');
  if (savedTheme) {
    document.body.className = savedTheme;
    DOM.themeSelect.value = savedTheme;
  }

  // Copy FEN
  DOM.btnCopyFen.addEventListener('click', () => {
    navigator.clipboard.writeText(DOM.fenInput.value).then(() => {
      DOM.btnCopyFen.textContent = '✓';
      setTimeout(() => {
        DOM.btnCopyFen.textContent = '📋';
      }, 1500);
    });
  });

  // Refresh metrics button in analysis tab
  DOM.btnRefreshMetrics.addEventListener('click', loadAnalysisData);
}

// ============================================================================
// 12. INITIAL BOOTSTRAP
// ============================================================================
window.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
  setupEventListeners();
  updateAgentInfoUI();
  // Auto-start initial game session
  startNewGame();
});
