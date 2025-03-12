# AI-Powered Student Attendance System

An intelligent attendance tracking system that uses facial recognition to automate student attendance in classes.

## Features

- Facial recognition-based attendance tracking
- Real-time attendance monitoring
- Class management system
- Student database
- Attendance reports and analytics
- User-friendly web interface

## Tech Stack

- Backend: Python with FastAPI
- Frontend: React with TypeScript
- Database: PostgreSQL
- AI: face-recognition library with dlib
- Authentication: JWT

## Prerequisites

- Python 3.8+
- Node.js 16+
- PostgreSQL
- Webcam access for facial recognition

## Setup Instructions

1. Clone the repository
2. Install backend dependencies:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```
4. Set up the database:
   ```bash
   # Create PostgreSQL database and update .env file
   ```
5. Start the application:
   ```bash
   # Backend
   cd backend
   uvicorn main:app --reload

   # Frontend
   cd frontend
   npm run dev
   ```

## Project Structure

```
.
├── backend/
│   ├── app/
│   ├── models/
│   ├── services/
│   └── ai/
├── frontend/
│   ├── src/
│   ├── public/
│   └── components/
└── docs/
```