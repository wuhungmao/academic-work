package Difficulties;

import boggle.BoggleGame;
import boggle.Position;

import java.util.ArrayList;
import java.util.Map;
import java.util.Random;

public class CMeasy extends ComputerMoveDifficulty {
    public int wordlimit = 20;
    public ArrayList<String> wordchosen = new ArrayList<>();
    public int letter_in_a_word = 5;
    public BoggleGame game;

    public CMeasy(BoggleGame game) {
        this.game = game;
    }

    public void bug() {
        Random rand = new Random();
        int numremove = rand.nextInt(0,this.wordlimit);
        while (numremove != 0 && !this.wordchosen.isEmpty()) {
            this.wordchosen.remove(0);
            numremove--;
        }
}

    @Override
    public void chooseRandomWords() {
        String word = "";
        int wordremain = this.wordlimit;
        for (Map.Entry<String, ArrayList<Position>> entry : this.game.allWords.entrySet()) {
            word = entry.getKey();
            if (!this.wordchosen.contains(word) && wordremain > 0 && word.length() <= letter_in_a_word) {
                this.wordchosen.add(word);
                wordremain--;
            }
            if (wordremain == 0) {
                break;
            }
        }
    }
    }
