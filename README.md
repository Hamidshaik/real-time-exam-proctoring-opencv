# Real-Time Exam Proctoring System (OpenCV)

A real-time webcam-based exam monitoring system built using OpenCV and Python.  
The system detects **face presence, multiple faces, head deviation (looking away), and prolonged eye invisibility** using classical computer vision techniques.

This project is designed as a **prototype** to demonstrate real-time monitoring logic, event-based violation tracking, and practical limitations of Haar Cascade–based vision systems.

> ⚠️ This is NOT a cheating detector.  
> It flags **suspicious behavior patterns** that may require human review.

---

## Features

- Real-time webcam monitoring
- Face detection using Haar Cascades
- Multiple face detection
- Head position–based attention tracking
- Eye visibility monitoring
- **Time-based event detection** (not frame-based)
- Event summary after session ends

---

## Why Time-Based Detection?

Most naive proctoring systems count violations per frame, which leads to massive false positives.

This system:
- Tracks how long a condition persists
- Counts an event **only if it exceeds a time threshold**
- Produces meaningful, reviewable summaries

---

## Tech Stack

- Python 3.x
- OpenCV
- NumPy

No deep learning models are used in this version.

---

## How It Works

1. Captures webcam frames in real time
2. Detects faces using Haar Cascades
3. Monitors:
   - No face present
   - Multiple faces present
   - Head deviation from screen center
   - Prolonged absence of detectable eyes
4. Flags events only when conditions persist beyond a set duration
5. Displays live status and generates a session summary

---

## Installation

```bash
pip install opencv-python numpy
