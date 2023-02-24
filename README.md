# mice-learning

## Example Commands

## Create a virtualenv

To run these command it is recommended to create a virtualenv to install all dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -e ../mice_base/
```

### Generating poisons

```bash
python3 find_poison_base.py \
        --out_dir ./poisons/ \
        --config sample-inputs/config.json
```
