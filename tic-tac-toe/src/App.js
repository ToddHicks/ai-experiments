import React, { useState } from "react";
import "./GameCard.css";

const API_BASE = "http://127.0.0.1:5009";

const TicTacToe = () => {
  const [gameId, setGameId] = useState(null);
  const [board, setBoard] = useState(Array(9).fill(0));
  const [winner, setWinner] = useState(null);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState({
    games_played: 0,
    wins: 0,
    losses: 0,
    ties: 0,
  });

  const startGame = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/new`, { method: "POST" });
      const data = await response.json();

      setGameId(data.game_id);
      setBoard(parseBoard(data.board));
      setWinner(null);
      const statsResponse = await fetch(`${API_BASE}/stats`, { method: "GET" });
      const statsData = await statsResponse.json();
      setStats({
        games_played: statsData.games_played,
        wins: statsData.last_10_wins,
        losses: statsData.last_10_losses,
        ties: statsData.last_10_ties,
      });
    } catch (error) {
      console.error("Error starting game:", error);
    }
    setLoading(false);
  };

  const makeMove = async (index) => {
    if (winner || board[index] !== 0) return;
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/act`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ game_id: gameId, action: index }),
      });

      const data = await response.json();
      setBoard(parseBoard(data.board));
      setWinner(data.winner);
    } catch (error) {
      console.error("Error making move:", error);
    }
    setLoading(false);
  };

  const parseBoard = (boardString) => {
    // This should be able to be cleaned up on the server side.
    return boardString
      .replace(/[\[\]\n]/g, "")
      .trim()
      .split(/\s+/)
      .map(Number);
  };

  return (
    <div className="game-container">
      <h1 className="game-title">Tic-Tac-Toe AI</h1>
      <p className="game-stats">
        Total Games: {stats.games_played}. Last 10 Games: {stats.wins} Wins,{" "}
        {stats.losses} Losses, {stats.ties} Ties.
      </p>
      {!gameId ? (
        <button
          className="custom-button"
          onClick={startGame}
          disabled={loading}
        >
          Start Game
        </button>
      ) : (
        <div className="game-card">
          <div className="game-card-content">
            {Array.from({ length: 3 }, (_, row) => (
              <div key={row} className="game-row">
                {board.slice(row * 3, row * 3 + 3).map((cell, col) => (
                  <button
                    key={row * 3 + col}
                    className="game-cell"
                    onClick={() => makeMove(row * 3 + col)}
                    disabled={cell !== 0 || winner !== null || loading}
                  >
                    {cell === 1 ? "X" : cell === -1 ? "O" : " "}
                  </button>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}
      {winner !== null && (
        <div className="game-result">
          <p>
            {winner === 1
              ? "AI Wins!"
              : winner === -1
              ? "You Win!"
              : "It's a Tie!"}
          </p>
          <button
            className="custom-button"
            onClick={startGame}
            disabled={loading}
          >
            New Game
          </button>
        </div>
      )}
    </div>
  );
};

export default TicTacToe;
