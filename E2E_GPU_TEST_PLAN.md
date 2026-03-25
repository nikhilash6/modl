# E2E GPU Test Plan — `--attach-gpu` with SDXL on Vast.ai

## Prerequisites
- [x] GPU bundle uploaded to R2 (`gpu-agent/modl-gpu-bundle.tar.gz`)
- [x] modl-cloud deployed with `hub.modl.run` as API base
- [x] `output_dir` fix committed on `feat/attach-gpu`
- [ ] Build modl binary from `feat/attach-gpu` branch
- [ ] Account has `gpu_access = true` in Postgres

## Setup

```bash
# Build the CLI from feat/attach-gpu
git checkout feat/attach-gpu
cargo build --release
# Use ./target/release/modl or alias it

# Verify auth works
modl auth status  # should show logged-in account

# Verify hub connectivity
curl -s https://hub.modl.run/health
```

## Test 1: Session provisioning (RTX 3090, ≤$0.25/hr)

```bash
modl gpu attach rtx3090 --idle 5m
```

**Expected:** Session created, polls Vast.ai, instance boots in 1-20 min.
**Watch for:**
- "No available rtx3090 instances" → Vast.ai filters too aggressive (relax reliability/dlperf)
- "GPU access is not yet available" → need `UPDATE accounts SET gpu_access = true WHERE ...`
- Timeout after 20 min → Vast.ai host issues, try again

```bash
modl gpu status  # check state transitions: provisioning → installing → ready
```

## Test 2: SSH into the instance

```bash
modl gpu ssh
```

**Check on the instance:**
```bash
# Is the agent running?
ps aux | grep modl
# Check agent logs
cat /tmp/modl-agent.log  # or check stdout

# Is modl binary there?
which modl
modl --version

# Is Python worker available?
ls /opt/modl/python/modl_worker/

# Are torch/diffusers installed?
python3 -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
python3 -c "import diffusers; print(diffusers.__version__)"

# Can we see the GPU?
nvidia-smi
```

## Test 3: Generate with SDXL

```bash
modl generate "a mountain landscape, oil painting" --base sdxl --attach-gpu --gpu-type rtx3090
```

**Expected flow:**
1. CLI connects to existing session (or provisions new one)
2. Submits job to orchestrator → agent picks it up
3. Agent pulls SDXL model (~6.5GB, first run only)
4. Agent runs generation via Python worker
5. Agent uploads artifact to R2
6. CLI downloads artifact from R2 to local output dir
7. Image saved locally

**Watch for:**
- "Failed to parse generate spec" → spec serialization issue
- Agent poll returns 204 forever → job not queued, check DB
- Model pull fails → check disk space, network on instance
- Python worker fails → SSH in, check `python3 -c "import diffusers"`
- Artifact upload fails → R2 presigned URL issue
- CLI can't download → presigned URL expired or wrong

## Test 4: Verify artifacts

```bash
# Check local output
ls -la ~/modl/outputs/
modl outputs list  # should show the generated image

# Check via modl gpu status
modl gpu status  # should show session is idle after job
```

## Test 5: Cleanup

```bash
modl gpu detach  # destroys the Vast.ai instance
modl gpu status  # should show "No active GPU sessions"
```

## Debugging

### Server-side (modl-cloud logs)
```bash
ssh dokku@91.99.16.140 logs modl-cloud -t  # tail logs
```

### Database queries
```sql
-- Check session state
SELECT id, state, gpu_type, price_per_hour, error_message FROM gpu_sessions ORDER BY created_at DESC LIMIT 5;

-- Check GPU jobs
SELECT id, session_id, job_type, status FROM gpu_jobs ORDER BY created_at DESC LIMIT 5;

-- Check events for a job
SELECT sequence, payload->>'event' FROM job_events WHERE job_id = '<job_id>' ORDER BY sequence;

-- Check artifacts
SELECT * FROM artifacts WHERE job_id = '<job_id>';

-- Enable GPU access for account
UPDATE accounts SET gpu_access = true WHERE email = '<your-email>';
```

### Vast.ai console
Check instance status at https://console.vast.ai/instances/

### Common fixes
- **Filters too strict:** In `shared/vastai.py`, try relaxing `reliability2 >= 0.95`, `dlperf >= 10`
- **Agent not starting:** SSH in, check if onstart script completed — look at `/var/log/` or Vast.ai console logs
- **Model pull interactive prompt:** Should be fixed (--variant fp16), but verify
- **R2 presigned URL fails:** Check R2 secrets on Dokku config
