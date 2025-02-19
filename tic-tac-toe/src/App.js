//import { Button } from "@/components/ui/button";
//import { Card, CardContent } from "@/components/ui/card";
import React, { useState } from "react";
import { Button } from "./components/ui/Button";
import { Card, CardContent } from "./components/ui/Card";

const API_BASE = "https://ai-experiments-1196.onrender.com";
//const API_BASE = "http://127.0.0.1:5009";

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
      console.log(data);
      setGameId(data.game_id);
      let fixedBoard = data.board
        .replace(/\n/g, " ") // Replace newlines with spaces
        .replace(/\s+/g, ",") // Replace spaces with commas
        .replace(/\[|]/g, "") // Remove square brackets
        .split(",") // Split into an array
        .map(Number); // Convert to numbers
      console.log(fixedBoard);
      setBoard(fixedBoard);
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
      console.log(data);

      // Ensure `data.board` is properly formatted
      let fixedBoard = data.board
        .replace(/[\[\]\n]/g, "") // Remove brackets and newlines
        .trim() // Remove leading/trailing spaces
        .split(/\s+/) // Split by whitespace
        .map(Number); // Convert to numbers

      console.log(fixedBoard);

      setBoard(fixedBoard.slice(0, 9));

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
          <CardContent className="flex flex-col gap-2">
            {Array(3)
              .fill(0)
              .map((_, row) => (
                <div key={row} className="flex gap-2">
                  {board.slice(row * 3, row * 3 + 3).map((cell, col) => (
                    <Button
                      key={row * 3 + col}
                      className="h-48 w-48 text-5xl flex items-center justify-center"
                      onClick={() => makeMove(row * 3 + col)}
                      disabled={cell !== 0 || winner !== null || loading}
                    >
                      {cell === 1 ? "X" : cell === -1 ? "O" : "\u00A0"}
                    </Button>
                  ))}
                </div>
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
          <Button onClick={startGame} disabled={loading}>
            Start Game
          </Button>
        </p>
      )}
    </div>
  );
};

export default TicTacToe;
