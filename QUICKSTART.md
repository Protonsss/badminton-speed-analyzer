# quick start guide

## for beginners

### step 1: start the backend

**option a - easy way (docker):**
```bash
cd backend
docker-compose up
```
wait till it says "ready to accept requests"

**option b - python way:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # windows: venv\Scripts\activate
pip install -r requirements.txt
python -m app.main
```

### step 2: open the frontend

just double-click `index-api.html`

or if that doesnt work:
```bash
python3 -m http.server 3000
```
then open browser to `http://localhost:3000/index-api.html`

### step 3: use it

1. click "choose video"
2. select your badminton video
3. click "start calibration"
4. click the 4 court corners in order (top-left, top-right, bottom-right, bottom-left)
5. click "analyze video"
6. wait (takes like 30 sec to 2 min depending on video length)
7. boom. your speed.

## troubleshooting

**"failed to connect to backend"**
- make sure backend is running on port 8000
- check `http://localhost:8000/api/v1/health` shows "healthy"

**"tracking failed" or "insufficient data"**
- shuttle wasnt visible enough in video
- try better lighting
- make sure shuttle is in frame most of the time
- film at higher fps if possible

**"calibration error"**
- make sure you clicked corners in right order
- corners should form a proper rectangle (not a line)
- all 4 corners need to be visible in frame

**slow processing**
- normal on cpu (5-10 fps)
- use gpu for faster processing (30+ fps)
- shorter videos process faster obviously

**inaccurate speeds**
- check your calibration points (most common issue)
- film perpendicular to court
- higher fps = more accurate
- make sure court dimensions are right (singles vs doubles)

## video requirements

**minimum:**
- 720p resolution
- 30 fps
- all 4 court corners visible
- shuttle visible most of the time

**recommended:**
- 1080p resolution
- 60 fps or higher
- good lighting
- white shuttle
- steady camera (tripod)

**ideal:**
- 1080p or 4k
- 120 fps or 240 fps
- professional lighting
- perpendicular camera angle

## example workflow

1. film your smash from the side of the court
2. make sure you can see all 4 corners
3. upload to the app
4. carefully click the 4 corners (this is the most important step)
5. hit analyze
6. wait for results
7. export csv or screenshot to share

## tips

- practice clicking the corners accurately. zoom in if needed.
- if speed seems way off, redo calibration
- film multiple attempts and pick the best one
- compare with friends to see whos faster
- keep videos under 10 seconds for faster processing

## whats next

once you got it working:
- try different camera angles to see what works best
- experiment with different fps
- compare speeds across different sessions
- train a custom yolo model for even better accuracy (advanced)

thats it. have fun tracking your smashes.

