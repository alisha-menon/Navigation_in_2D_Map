# hd_recall_2d_nav

Requires numpy and pygame packages

Run "play_game.py" and follow prompts to choose gameplay type.
If playing the game live, data will be written to "game_dat.out". Format of output data is CSV where each line represents one action:

LSENSOR, RSENSOR, USENSOR, DSENSOR, ACTION

The first four numbers are binary flags for the left, right, up, down sensors 
respectively. ACTION is an integer between 0 and 3 where:  
    0 - move left  
    1 - move right  
    2 - move up  
    3 - move down  

