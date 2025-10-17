# dev notes

random notes about building this thing

## architecture decisions

### why fastapi over flask
- automatic openapi docs
- pydantic validation built in
- async support (even tho we dont use it much yet)
- faster than flask according to benchmarks
- nicer api with type hints

### why not tensorflow
tried tensorflow first but pytorch just felt cleaner. ultralytics yolo is pytorch-based anyway so made sense to stick with it.

### why separate detector and tracker
initially had everything in one class but it got messy. separating concerns made testing easier and lets you swap detection methods without touching tracking logic.

### kalman vs particle filter
tried both. kalman was way faster and worked fine for this use case. particle filter was overkill and slower.

## bugs i fixed

### calibration coordinate mapping bug
big one. was using canvas coordinates directly but forgot the video gets letterboxed in the canvas. so calibration points were off by like 20-30%. added `canvasToVideoCoords` to properly map click positions to actual video pixels.

### optical flow drift
optical flow would drift over time. fixed by re-locking with yolo every N frames. also added kalman filter which helped a ton with smoothing.

### savgol filter edge effects
savgol was giving weird values at the start and end of trajectory. now i just make sure window size isnt larger than data length and use appropriate polynomial order.

### yolo memory leak
yolo was accumulating memory over long videos. turned out i wasnt releasing the cv2 videocapture properly. added try/finally to ensure cleanup.

## performance optimizations

- only run yolo every N frames instead of every frame (8 frame interval works well)
- use yolov8n (nano) instead of larger models for speed
- optical flow is way faster than detection so use it between detections
- letterbox resize for yolo instead of direct resize (preserves aspect ratio, better accuracy)

## things i learned

### homography is finicky
small errors in corner clicks = big errors in coordinate transformation. thats why calibration validation is so important.

### shuttlecock aerodynamics are weird
tried linear drag model first but it didnt fit. quadratic drag model works way better: `F = -k*v²`. makes sense cuz shuttle has weird shape.

### fps matters way more than resolution
tried 4k 30fps vs 1080p 120fps. 1080p 120fps was way more accurate. time resolution > spatial resolution for speed measurement.

### kalman tuning
measurement noise = how much you trust your observations
process noise = how much you expect the state to change
took a while to find good values. too low measurement noise = jumpy tracking. too high = laggy tracking.

## todos (maybe)

- [ ] async processing with celery or rq
- [ ] postgres for storing results
- [ ] user accounts / auth
- [ ] batch processing multiple videos
- [ ] real-time camera feed processing
- [ ] mobile app (react native?)
- [ ] better error messages in frontend
- [ ] progress bar during processing
- [ ] shuttle trajectory heatmap visualization
- [ ] compare multiple videos side by side
- [ ] export to video with overlay

## random observations

- most people film at like 30fps which isnt great for this
- iphone slow-mo (240fps) works really well
- tracking fails way more often on old videos with bad lighting
- calibration is the hardest part for users. needs better ui.
- training custom yolo would help a lot but need labeled data

## testing notes

should add more tests:
- integration test with actual video file
- test with various video formats
- stress test with really long videos
- test calibration edge cases (super skewed angles, etc)

coverage is like 60% right now which is ok but could be better.

## deployment thoughts

current docker setup works but for real production would want:
- kubernetes for scaling
- redis for job queue
- s3 for video storage (dont store locally)
- cloudfront cdn
- sentry for error tracking
- datadog/newrelic for monitoring
- automated backups
- ci/cd pipeline (github actions)

but thats overkill for personal use.

## code smell i havent fixed

- global processor instance in routes.py is kinda janky
- should probably use dependency injection properly
- logging could be more structured (json logs)
- some functions are too long (processor.process_video)
- not enough input validation in some places
- error messages could be more helpful

but hey it works so ¯\_(ツ)_/¯

## interesting alternatives i considered

**tracking:**
- SORT/DeepSORT (overkill for single object)
- CSRT tracker (opencv builtin, but slower than optical flow)
- correlation filters (tried it, wasnt great)

**detection:**
- haar cascades (lol no)
- color-based detection (too unreliable)
- template matching (nope)
- trained svm (too old school)

**speed calculation:**
- polynomial fit (tried 2nd, 3rd order, didnt work as well as physics model)
- simple moving average (too much smoothing, loses peak)
- exponential smoothing (ok but savgol is better)

## if i built this again

would probably:
- use typescript instead of python for backend (or at least add mypy from start)
- set up proper dependency injection from day 1
- write tests alongside code instead of after
- use sqlalchemy from start even if not needed yet
- better logging structure (json logs, correlation ids)
- document api with more examples

but overall happy with how it turned out.

## useful resources

- opencv docs (always open)
- ultralytics yolo docs
- filterpy documentation
- fastapi docs (really good)
- stackoverflow (obviously)

## hardware tested on

- macbook pro m1 (fast as hell, ~40fps processing)
- windows desktop i7 + rtx 3060 (~30fps processing)
- linux server cpu only (~8fps, slow but works)
- raspberry pi 4 (lol dont. took forever.)

---

last updated: when i finished the refactor

