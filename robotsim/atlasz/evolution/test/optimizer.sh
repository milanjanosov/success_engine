
#!/bin/bash
echo "First arg: $1"
echo "First arg: $2"
echo "First arg: $3"
echo "First arg: $4"

screen -S $1 bash -c "source /opt/virtualenv-python2.7/bin/activate; python2.7 test_evolution.py -m  $2 -min $3 -max $4 "
