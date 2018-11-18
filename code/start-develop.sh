#!/bin/bash -e

DEPENDENCIES="virtualenv python3 pip3"
DEP_SATISFY=1
echo "Checking dependencies:"
for prog in ${DEPENDENCIES}
do
  if [[ -z $(which ${prog}) ]]; then
    echo "${prog} not installed!"
    DEP_SATISFY=0
  else
    echo "${prog} installed!"
  fi
done

if [[ $DEP_SATISFY == 0 ]]; then
  echo "Dependencies not satisfied!"
  exit 1
else
  echo ""
fi

echo "Creating virtualenv:"
virtualenv -p $(which python3) venv
echo ". ./venv/bin/activate" >> .env
echo "deactivate" >> .out
echo "deactivate" >> .env.leave
echo ""

echo "Installing Python dependencies:"
. .env
pip install -r requirements.txt
. .out
echo ""

echo "Done!"
