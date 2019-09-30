#download dataset
wget --load-cookies ./cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XZm5L27eQasrvMq008l6vW8I8ysnavSM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XZm5L27eQasrvMq008l6vW8I8ysnavSM" -O CrowdHuman_train01.zip && rm -rf ./cookies.txt
wget --load-cookies ./cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1KgreKkhPfIiZHkl-x5K7p5LHKsfZss5q' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1KgreKkhPfIiZHkl-x5K7p5LHKsfZss5q" -O CrowdHuman_train02.zip && rm -rf ./cookies.txt
wget --load-cookies ./cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Af1rBvQSxOmXphoNNtKzAJyrEeeAF-aV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Af1rBvQSxOmXphoNNtKzAJyrEeeAF-aV" -O CrowdHuman_train03.zip && rm -rf ./cookies.txt
wget --load-cookies ./cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dRRL6eKE1v_1Kb_R8nZGhTQ0HzYZ-Pss' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dRRL6eKE1v_1Kb_R8nZGhTQ0HzYZ-Pss" -O CrowdHuman_val.zip && rm -rf ./cookies.txt
wget --load-cookies ./cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uBB3psTLteVEP2Wg466DsYVx1eV8LacU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uBB3psTLteVEP2Wg466DsYVx1eV8LacU" -O annotation_train.odgt && rm -rf ./cookies.txt
wget --load-cookies ./cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=151-MHsdCni1izANEuZA3q3Pp7Jwpi-PX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=151-MHsdCni1izANEuZA3q3Pp7Jwpi-PX" -O annotation_val.odgt && rm -rf ./cookies.txt
wget --load-cookies ./cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GYzUB07J35P5Y_DJCwLzJt72x_Mkf9Yl' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GYzUB07J35P5Y_DJCwLzJt72x_Mkf9Yl" -O CrowdHuman_test.zip && rm -rf ./cookies.txt
mkdir images
mkdir annotations

unzip '*.zip'
mv Images/* images/
mv images_test/* images/
mv annotation_train.odgt annotations/
mv annotation_val.odgt annotations/

rm -rf Images
rm -rf images_test
python crowd_to_coco.py
