package Difficulties;

import boggle.BoggleGame;
import boggle.Position;

import java.util.ArrayList;
import java.util.Map;

public class CMMedium extends ComputerMoveDifficulty {
    public int wordlimit = 30;
    public ArrayList<String> wordchosen = new ArrayList<>();
    public int totalscore;
    public int letter_in_a_word = 7;
    public BoggleGame game;


    public CMMedium(BoggleGame game) {
        this.game = game;
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
