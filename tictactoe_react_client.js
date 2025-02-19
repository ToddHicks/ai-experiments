import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import React, { useState } from "react";

const API_BASE = "https://ai-experiments-1196.onrender.com";

const TicTacToe = () => {
  const [gameId, setGameId] = useState(null);
  const [board, setBoard] = useState(Array(9).fill(0));
  const [winner, setWinner] = useState(null);
  const [loading, setLoading] = useState(false);

  const startGame = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/new`, { method: "POST" });
      const data = await response.json();
      setGameId(data.game_id);
      setBoard(JSON.parse(data.board));
      setWinner(null);
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
      setBoard(JSON.parse(data.board));
      setWinner(data.winner);
    } catch (error) {
      console.error("Error making move:", error);
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col items-center p-4">
      <h1 className="text-xl font-bold mb-4">Tic-Tac-Toe AI</h1>
      {!gameId && (
        <Button onClick={startGame} disabled={loading}>
          Start Game
        </Button>
      )}
      {gameId && (
        <Card className="p-4 mt-4">
          <CardContent className="grid grid-cols-3 gap-2">
            {board.map((cell, index) => (
              <Button
                key={index}
                className="h-16 w-16 text-xl"
                onClick={() => makeMove(index)}
                disabled={cell !== 0 || winner !== null || loading}
              >
                {cell === 1 ? "X" : cell === -1 ? "O" : ""}
              </Button>
            ))}
          </CardContent>
        </Card>
      )}
      {winner !== null && (
        <p className="mt-4 text-lg font-bold">
          {winner === 1
            ? "AI Wins!"
            : winner === -1
            ? "You Win!"
            : "It's a Tie!"}
        </p>
      )}
    </div>
  );
};

export default TicTacToe;
