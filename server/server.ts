import express from 'express';

const express = require('express');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 3000;

// Отдаем статические файлы из папки 'public'
app.use(express.static(path.join(__dirname, 'public')));

// Отправляем index.html на все остальные запросы
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Сервер запущен на http://localhost:${PORT}`);
});