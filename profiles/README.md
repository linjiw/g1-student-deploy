# Deploy profiles

A deploy profile names everything a run of the student depends on besides the robot: the planner bundle, its deploy
rules plugin, the guards, the walking lookahead, the lidar blind and the planner options. The robot's own facts
(interface, lidar, body, mount calibration, odometry) stay in `robots/<name>.json`.

```bash
./g1 sim --profile deploy --layout lab-door-aligned --script "..."
./g1 robot --robot lab-g1 --profile deploy --goal 2.0 0
./g1 sim --profile deploy --rear-blind-deg 0 ...     # a command-line option wins over the profile
```

Precedence: command line > deploy profile > robot profile > the console's defaults. `--profile` takes a name
(`profiles/<name>.json`, then `private/profiles/<name>.json`) or a path to a `.json` file. Relative `bundle` and
`rules` paths are relative to this repository.

| Key | Console option | Values |
|---|---|---|
| `bundle` | `--bundle` | planner bundle directory |
| `rules` | `--rules` | deploy rules plugin (`.py`), or null |
| `guards` | `--guards` | `off`, `deploy`, `walls` |
| `goal_lookahead` | `--goal-lookahead` | metres (1.0 = slow walking) |
| `rear_blind_deg`, `rear_blind_range_m` | `--rear-blind-deg`, `--rear-blind-range-m` | 180 and 1.0 = near-rear blind; 0 = none |
| `debug_plans` | `--debug-plans` | dump every plan's inputs and result (needed for replays) |
| `planner_impl`, `planner_precision` | `--planner-impl`, `--planner-precision` | `fused` / `graph` / `eager`; `fp32` / `tf32` / `bf16` |
| `candidates`, `ode_steps`, `self_margin` | same names | planner K, ODE steps, self-filter margin (m) |

Every session's `session.json` records the profile (name, file, SHA-256, the options the command line overrode) and
a **deploy identity**: one hash over the effective settings, the bundle MANIFEST and the rules plugin. Sessions with
the same identity ran the same student the same way. Profiles for bundles kept outside this repository go in
`private/profiles/` (git-ignored).

| Profile | What it is |
|---|---|
| `deploy` | The robot's current setup (DEPLOY_KIT.md §11): deployed bundle, deploy guards, lookahead 1.0, near-rear blind |
| `deploy-full-lidar` | The same with the whole lidar fused |
