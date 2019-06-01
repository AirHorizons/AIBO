runtwo: 
	python run_sc2minigame.py \
                         --bot1=bots.nc_example_v6.drop_bot.DropBot \
			 --bot2=bots.nc_example_v5.reaper_bot.ReaperBot \
                         --realtime=False

runReaperbot:
	python run_sc2minigame.py \
			--bot1=bots.nc_example_v5.reaper_bot.ReaperBot

runDrop:
	python run_sc2minigame.py \
			--bot1=bots.nc_example_v6.drop_bot.DropBot \
			--realtime=False
