To get started please run:

git lfs install

git clone https://github.com/ahgeller/chess-attack-analysis

Prerequisites:
pip install python-chess pandas matplotlib numpy tqdm

Reproduction: 
Your CSV file should have these columns:
id,moves,opening_name,white_rating,black_rating
1,"e4 e5 Nf3 Nc6 Bb5 a6","Ruy Lopez",1500,1480
2,"d4 d5 c4 e6 Nc3 Nf6","Queen's Gambit Declined",1620,1595

For your convienence ive converted the files: games.csv  into TESTatcData.csv 

**Processing Time Warning**: If you would like to run the game files through 
attack_value_full.py to calculate attack values, I recommend only using games.csv 
as the lichess file would require millions of iterations and days of processing time. 
If you want to use it, remove the # to enable .take(range(100000)) (or adjust the number to whatever amount you prefer).

However, I've already processed the entire lichess-08-2014.csv file for you! 
If you would simply like to see the results, look at the graph folder.

Unzip the CSV files.7z

After this step, run run.py.

It will create/recreate the TESTatcData.csv and TESTatcDataLICHESSS.csv data.

Finally run "whatdoesthistellme.py" to create the graphs I've already made.

opening_attack_advantage.png - Which openings create attacking chances

white_vs_black_correlation.png - Attack symmetry analysis

rating_vs_attack.png - Attack patterns by skill level

attack_progression.png - Attack values throughout the game

attack_distributions.png - Statistical distributions of attack values

attack_vs_outcome.png - How attack correlates with winning

<img width="1500" height="1050" alt="opening_attack_advantage" src="https://github.com/user-attachments/assets/d72cceeb-4014-4ef8-a82a-08088ab30bfb" />

<img width="1200" height="1200" alt="white_vs_black_correlation" src="https://github.com/user-attachments/assets/c7557617-f19e-4456-9a65-5da4c35fbb4b" />

<img width="2100" height="750" alt="rating_vs_attack" src="https://github.com/user-attachments/assets/aa45cb07-3a1b-4597-abae-e850f9e94b81" />

<img width="1800" height="750" alt="attack_progression" src="https://github.com/user-attachments/assets/88a555e7-c86f-4146-9888-ada7b781e5f3" />

<img width="2250" height="750" alt="attack_distributions" src="https://github.com/user-attachments/assets/3a3bcaa6-6d1a-45b3-8cc9-7c0f9c140150" />

<img width="2100" height="750" alt="attack_vs_outcome" src="https://github.com/user-attachments/assets/352b865f-60ad-4174-9803-ee3ddbb8fd84" />
