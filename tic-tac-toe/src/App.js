import React, { useEffect, useState } from "react";
import "./GameCard.css";

const API_BASE = "https://ai-experiments-1196.onrender.com";

const TicTacToe = () => {
  const [gameId, setGameId] = useState(null);
  const [board, setBoard] = useState(Array(9).fill(0));
  const [winner, setWinner] = useState(null);
  const [loading, setLoading] = useState(false);
  const [logic, setLogic] = useState(null);
  const [stats, setStats] = useState({
    games_played: 0,
    wins: 0,
    losses: 0,
    ties: 0,
  });

  useEffect(() => {
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
  }, []);

  const startGame = async () => {
    setLoading(true);
    setLogic(null);
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
    setLogic(null);

    try {
      const response = await fetch(`${API_BASE}/act`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ game_id: gameId, action: index }),
      });

      const data = await response.json();
      setBoard(parseBoard(data.board));
      if (data.winner != null) {
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

  const getLogic = async () => {
    if (!gameId) return;
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/matrix`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ game_id: gameId }),
      });

      const data = await response.json();
      setLogic(data);
    } catch (error) {
      console.error("Error fetching logic:", error);
    }
    setLoading(false);
  };

  const parseBoard = (boardString) => {
    return boardString
      .replace(/[\[\]\n]/g, "")
      .trim()
      .split(/\s+/)
      .map(Number);
  };

  return (
    <div className="game-container">
      <h1 className="game-title">Tic-Tac-Toe Learning AI</h1>
      <p className="game-stats">
        Total Games: {stats.games_played} <br />
        AI Stats for the last 10 games <br />
        Wins: {stats.wins} <br />
        Losses: {stats.losses} <br />
        Ties: {stats.ties}
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
          <button
            className="custom-button"
            onClick={getLogic}
            disabled={loading}
          >
            Get Logic
          </button>
          {logic && (
            <pre className="game-logic">{JSON.stringify(logic, null, 2)}</pre>
          )}
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
