package Difficulties;


import boggle.BoggleGame;

import java.security.PublicKey;
import java.util.Random;
import java.util.ArrayList;
import java.util.Map;


public abstract class ComputerMoveDifficulty {
    public int wordlimit;
    public ArrayList<String> wordchosen;
    public int totalscore;
    public int letter_in_a_word;
    public BoggleGame game;

    public abstract void chooseRandomWords();
    }
