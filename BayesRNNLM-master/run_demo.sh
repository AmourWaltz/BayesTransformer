python main.py --train --data=data/ptb --model lstm --gate-type bayes --position 0 --save demo/ptblstmBayesscratch --no-tied --cuda 1 
python main.py --train --data=data/ptb --model lstm --gate-type bayes --position 1 --save demo/ptblstmBayesP1scratch --no-tied --cuda 1 
python main.py --train --data=data/ptb --model lstm --gate-type bayes --position 2 --save demo/ptblstmBayesP2scratch --no-tied --cuda 1 
python main.py --train --data=data/ptb --model lstm --gate-type bayes --position 3 --save demo/ptblstmBayesP3scratch --no-tied --cuda 1 
python main.py --train --data=data/ptb --model lstm --gate-type bayes --position 4 --save demo/ptblstmBayesP4scratch --no-tied --cuda 1 