import React, { useState } from "react";
import "./GameCard.css";

const API_BASE = "https://ai-experiments-connect4-server.onrender.com";

const Connect4 = () => {
  const [gameId, setGameId] = useState(null);
  const [board, setBoard] = useState(
    Array(6)
      .fill()
      .map(() => Array(7).fill(0))
  );
  const [winner, setWinner] = useState(null);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState({
    games_played: 0,
    wins: 0,
    losses: 0,
    ties: 0,
  });
  /*useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch(`${API_BASE}/stats`, { method: "GET" });
        const data = await response.json();
        setStats({
          games_played: data.games_played,
          wins: data.last_10_wins,
          losses: data.last_10_losses,
          ties: data.last_10_ties,
        });
      } catch (error) {
        console.error("Error fetching stats:", error);
      }
    };

    fetchStats();
  }, []);*/

  const startGame = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/new_connect4`, {
        method: "POST",
      });
      const data = await response.json();
      setGameId(data.game_id);
      setBoard(data.board);
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

  const makeMove = async (col) => {
    if (winner || loading) return;
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/act_connect4`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ game_id: gameId, action: col }),
      });
      const data = await response.json();
      setBoard(data.board);
      if (data.winner !== null) {
        setWinner(data.winner);
        const statsResponse = await fetch(`${API_BASE}/stats`, {
          method: "GET",
        });
        const statsData = await statsResponse.json();
        setStats({
          games_played: statsData.games_played,
          wins: statsData.last_10_wins,
          losses: statsData.last_10_losses,
          ties: statsData.last_10_ties,
        });
      }
    } catch (error) {
      console.error("Error making move:", error);
    }
    setLoading(false);
  };

  return (
    <div className="game-container">
      <h1 className="game-title">Connect4 AI Game</h1>
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
          {board.map((row, rowIndex) => (
            <div key={rowIndex} className="game-row">
              {row.map((cell, colIndex) => (
                <button
                  key={colIndex}
                  className="game-cell"
                  onClick={() => makeMove(colIndex)}
                  disabled={cell !== 0 || winner !== null || loading}
                >
                  {cell === 1 ? "🔴" : cell === -1 ? "🔵" : "⚪"}
                </button>
              ))}
            </div>
          ))}
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

export default Connect4;
