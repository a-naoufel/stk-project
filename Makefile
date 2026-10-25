PYTHON := .venv/bin/python
PYSTK2 := .venv/bin/pystk2

.PHONY: setup test check race dashboard train smoke evaluate package
setup:
	./scripts/uv sync --frozen
test:
	$(PYTHON) -m unittest discover -s tests -v
check:
	$(PYTHON) scripts/check_submission.py --track lighthouse --steps 100
race:
	$(PYSTK2) race --hide --track lighthouse --num-karts 4 --max-steps 1500 stk_actor
dashboard:
	@echo "Dashboard starts PAUSED. Once ready, open http://localhost:8050 and click Run."
	$(PYSTK2) race --hide --web --track lighthouse --num-karts 4 stk_actor
train:
	$(PYTHON) -m stk_actor.learn --steps 100000
smoke:
	$(PYTHON) -m stk_actor.learn --steps 256 --n-steps 128 --batch-size 64 --track lighthouse --output outputs/smoke/pystk_actor.pth
evaluate:
	$(PYTHON) -m scripts.evaluate --agent deployed
package:
	$(PYTHON) -m scripts.package_submission
