PY=python
.PHONY: install test validate data select train tune constant run figures all clean
install:  ; pip install -e .
test:     ; pytest -q
validate: ; $(PY) scripts/00_validate_sim.py
data:     ; $(PY) scripts/01_collect.py
select:   ; $(PY) scripts/02_select_metrics.py
train:    ; $(PY) scripts/03_train.py
tune:     ; $(PY) scripts/04_tune_baselines.py
constant: ; $(PY) scripts/07_exp_constant_rr.py
run:      ; $(PY) scripts/05_run_all.py
figures:  ; $(PY) scripts/06_make_figures.py
all: validate data select train tune constant run figures
clean:    ; rm -rf results/raw/* results/figures/* results/tables/* results/models/*
