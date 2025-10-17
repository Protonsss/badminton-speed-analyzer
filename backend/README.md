# badminton speed thing

ok so basically i got tired of not knowing how fast my smashes were so i built this. uses pytorch and yolo to track the shuttle and calculate speed. its pretty accurate if you film it right.

## what it does

- takes your badminton video
- you click 4 corners of the court (for calibration)
- it finds the shuttle using yolo v8
- tracks it frame by frame with optical flow
- uses kalman filter to smooth out the noise
- calculates speed in real meters/second using the court calibration
- fits a physics model (quadratic drag) to get the initial smash speed before air resistance slows it down

## how to run this

```bash
cd backend

# make virtual env
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on windows

# install stuff
pip install -r requirements.txt

# run the server
python -m app.main
```

server starts on localhost:8000

## using docker instead

if you dont wanna deal with python setup:

```bash
docker-compose up
```

thats it. hits port 8000.

## frontend

open `index-api.html` in browser. upload video, calibrate the court, hit analyze. 

or run a local server:
```bash
python -m http.server 3000
```
then go to localhost:3000/index-api.html

## how accurate is this actually

depends on your camera tbh:
- 60fps video: ±5-8 km/h error
- 120fps: ±3-5 km/h 
- 240fps: ±2-3 km/h (if ur fancy)

filming tips that actually matter:
- put camera perpendicular to the court side, not at an angle
- 2-3m high
- all 4 court corners visible in frame
- decent lighting, no shadows
- keep it steady (tripod if u got one)
- white shuttle shows up better

## the code structure

```
backend/
├── app/
│   ├── core/              # where the actual work happens
│   │   ├── detector.py    # yolo detection
│   │   ├── tracker.py     # optical flow + kalman
│   │   ├── calibration.py # court math
│   │   ├── speed_calculator.py  # speed analysis
│   │   └── processor.py   # ties everything together
│   ├── api/               # rest api stuff
│   │   ├── routes.py      # endpoints
│   │   └── schemas.py     # request/response models
│   ├── config.py          # settings
│   └── main.py            # fastapi app
├── tests/                 # pytest tests
└── requirements.txt
```

## how it actually works under the hood

### court calibration

so you click 4 corners of the court in the video. the code uses these points to create a homography matrix - basically a transformation that maps pixels in your video to real-world meters on the court. singles court is 13.4m × 5.18m, doubles is 13.4m × 6.10m.

the tricky part was making sure the points are in the right order and not all in a line (collinear). i added validation to check the area of the quad and cross products to make sure its actually a proper quadrilateral.

### shuttle detection

two approaches here:

**yolo v8**: downloads yolov8n.pt (nano model) first time you run it. runs inference on the frame, looks for objects, returns bounding boxes with confidence scores. i use the center of the bbox as the shuttle position. runs every N frames (default 8) to save compute.

**motion detector (fallback)**: if yolo fails or you dont want to use it, theres a backup detector that uses background subtraction. basically diffs the current frame with previous, threshold it, does some morphological ops (open + dilate) to clean up noise, finds contours, filters by size, picks the brightest one (shuttle is white). works ok but not as good as yolo.

### tracking

this is where it gets interesting. i use a hybrid approach:

1. **optical flow**: lucas-kanade pyramidal optical flow. takes a point from previous frame and finds where it moved to in current frame. fast and works great for frame-to-frame tracking but can drift over time.

2. **kalman filter**: constant velocity model with 4 states [x, y, vx, vy]. smooths out the noisy measurements from optical flow. the filter predicts where the shuttle should be, then updates when it gets a new measurement. tuned the process noise and measurement noise to work with shuttle dynamics.

3. **re-locking**: every 8 frames, runs yolo again to get a fresh detection and reset the tracker. prevents drift accumulation.

so the flow is: yolo detect → optical flow track → kalman smooth → repeat

### speed calculation

once i have trajectory points in meters (x, y, t), calculating speed is straightforward: distance / time between consecutive points.

but raw speeds are noisy af so i use savitzky-golay filter (scipy). its basically a moving polynomial fit - smooths the data while preserving peaks better than a moving average would.

### physics model

the cool part: i fit a quadratic drag model to estimate the initial smash speed before air slows it down.

physics says: `dv/dt = -k*v²` (drag proportional to velocity squared)

solution: `v(t) = v₀ / (1 + k*v₀*t)`

i use scipy.optimize.least_squares to fit this model to the measured speeds. gives you:
- v₀: initial speed (what you actually want)
- k: drag coefficient
- fit error: how well the model matches reality
- r²: goodness of fit

usually get r² > 0.8 which is pretty good for real-world data.

## api endpoints

### POST /api/v1/upload
upload video file. returns file_id to use later.

### POST /api/v1/calibrate
```json
{
  "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "court_type": "singles"
}
```

### POST /api/v1/analyze
processes the video. takes a while depending on video length.
```
file_id: <from upload>
use_yolo: true
start_frame: 0
end_frame: null
```

returns json with trajectory, speeds, etc.

### GET /api/v1/health
healthcheck. returns status + cuda availability.

## config

copy `env.example` to `.env` and change stuff:

- `YOLO_MODEL_PATH`: path to custom trained model (or leave empty for default)
- `YOLO_DEVICE`: cuda or cpu
- `RELOCK_INTERVAL`: how often to run yolo (lower = more accurate but slower)
- `SMOOTH_WINDOW`: savgol filter window size
- `KALMAN_MEASUREMENT_NOISE`: tune if tracking is jumpy
- `KALMAN_PROCESS_NOISE`: tune if tracking lags behind

## custom yolo model (advanced)

the default yolo model isnt trained on shuttlecocks so accuracy could be better. if you wanna train your own:

1. collect like 1000-5000 frames with shuttles
2. label them with bounding boxes (use roboflow or labelimg)
3. train yolo:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='shuttle.yaml', epochs=100, imgsz=640)
```
4. put the path in config: `YOLO_MODEL_PATH=path/to/best.pt`

ive seen people get 95%+ mAP with good training data.

## testing

```bash
pytest
```

i wrote some basic tests for calibration and api endpoints. coverage isnt 100% but hits the main stuff.

## known issues / limitations

- tracking fails if shuttle goes behind the net (occlusion)
- struggles with very fast shots at low fps
- motion detector gets confused by player movement in frame
- calibration assumes planar court (works fine unless ur filming from a weird angle)
- no multi-shuttle handling (assumes one shuttle in frame)

## performance

on my machine (i7 + rtx 3060):
- yolo inference: ~30-40 fps
- optical flow: ~100+ fps
- overall processing: ~20-30 fps realtime

cpu only is like 5-10 fps.

## why pytorch instead of just opencv

tried opencv dnn module first but yolo performance was way worse. pytorch + ultralytics just works better and the code is cleaner. also makes it easy to train custom models if needed.

## deployment

the docker setup is production-ready. uses uvicorn with multiple workers, health checks, proper logging, etc.

for real deployment id add:
- redis for job queue (async processing)
- postgres to store results
- s3 for video storage
- nginx reverse proxy
- let's encrypt ssl
- prometheus + grafana for monitoring

but for personal use the current setup works fine.

## credits

- yolov8 by ultralytics (goated)
- opencv for video/image processing
- filterpy for kalman filter
- scipy for signal processing and optimization
- fastapi for the api (way better than flask)

## license

MIT or whatever idc use it however u want

## final notes

this was a fun project to build. learned a lot about computer vision and tracking. the physics model fitting was probably the coolest part - actually getting real physics to match measured data is satisfying.

if u have questions or find bugs dm me or open an issue i guess

also if u train a good shuttle detection model lmk id love to use it

peace ✌️
