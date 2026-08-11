# Structural stages {#stage_structural}

Stages that restructure the dataflow graph (concatenation, splitting) rather than
transform or compress data.

| Stage | Description |
|---|---|
| \subpage stage_merge | Concatenate N producer ports into one buffer (forward) / split back (inverse) |
| \subpage stage_roibin_split | Split a field into full-resolution ROI boxes + a (optionally binned) background, so each branch can carry its own error bound |
