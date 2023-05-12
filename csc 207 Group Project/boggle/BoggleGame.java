package boggle;

import Difficulties.CMMedium;
import Difficulties.CMeasy;
import Difficulties.CMhard;
import javafx.scene.control.TextField;
import timer.GameTimer;

import java.util.*;

/**
 * The specific round of a game
 */
public class BoggleGame {
    /**
     * stores game statistics
     */ 
    private final BoggleStats gameStats;

    private final Dictionary dict;

    private final BoggleGrid grid;

    public Map<String,ArrayList<Position>> allWords;

    private int time; // Time for creating the timer， initialized when the timer is created for the first time

    protected GameTimer timer;

    protected boolean paused = false;

    /**
     * BoggleGame constructor
     */
    public BoggleGame(int size, BoggleStats stats) {
        gameStats = stats;
        grid = new BoggleGrid(size);
        dict = new Dictionary("wordlist.txt"); //you may have to change the path to the wordlist, depending on where you place it.
        allWords = new HashMap<>();
        findAllWords();
    }


    /*
     * This should be a recursive function that finds all valid words on the boggle board.
     * Every word should be valid (i.e. in the dict) and of length 4 or more.
     * Words that are found should be entered into the allWords HashMap.  This HashMap
     * will be consulted as we play the game.
     *
     * Note that this function will be a recursive function.  You may want to write
     * a wrapper for your recursion. Note that every legal word on the Boggle grid will correspond to
     * a list of grid positions on the board, and that the Position class can be used to represent these
     * positions. The strategy you will likely want to use when you write your recursion is as follows:
     * -- At every Position on the grid:
     * ---- add the Position of that point to a list of stored positions
     * ---- if your list of stored positions is >= 4, add the corresponding word to the allWords Map
     * ---- recursively search for valid, adjacent grid Positions to add to your list of stored positions.
     * ---- Note that a valid Position to add to your list will be one that is either horizontal, diagonal, or
     *      vertically touching the current Position
     * ---- Note also that a valid Position to add to your list will be one that, in conjunction with those
     *      Positions that precede it, form a legal PREFIX to a word in the Dictionary (this is important!)
     * ---- Use the "isPrefix" method in the Dictionary class to help you out here!!
     * ---- Positions that already exist in your list of stored positions will also be invalid.
     * ---- You'll be finished when you have checked EVERY possible list of Positions on the board, to see
     *      if they can be used to form a valid word in the dictionary.
     * ---- Food for thought: If there are N Positions on the grid, how many possible lists of positions
     *      might we need to evaluate?
     *
     * @param allWords A mutable list of all legal words that can be found, given the grid letters
     * @param dict A dictionary of legal words
     * @param grid A boggle grid, with a letter at each position on the grid
     */
    private void findAllWords() {
        for (int r = 0; r < grid.numRows(); r++) {
            for (int c = 0; c < grid.numCols(); c++) {
                allWords.putAll(r_helper(new ArrayList<>(List.of(new Position(r, c))),
                        Character.toString(grid.getCharAt(r, c))));
            }
        }
    }

    private Map<String,ArrayList<Position>> r_helper(ArrayList<Position> p, String s) {
        Map<String,ArrayList<Position>> m = new HashMap<>();
        if (!dict.isPrefix(s)) {
            return m;
        }
        if (s.length() > 2 && dict.containsWord(s)) {
            m.put(s.toUpperCase(), p);
        }
        for (int row = Math.max(0, p.get(p.size() - 1).getRow(0) - 1); row < Math.min(grid.numRows(),
                p.get(p.size() - 1).getRow(0) + 2); row++) {
            for (int col = Math.max(0, p.get(p.size() - 1).getCol(0) - 1); col < Math.min(grid.numCols(),
                    p.get(p.size() - 1).getCol(0) + 2); col++) {
                boolean dup = false;
                for (Position pos : p) {
                    if (pos.getRow(0) == row && pos.getCol(0) == col) {
                        dup = true;
                        break;
                    }
                }
                if (dup) { continue; }
                ArrayList<Position> n = new ArrayList<>(p);
                n.add(new Position(row, col));
                m.putAll(this.r_helper(n, s.concat(Character.toString(grid.getCharAt(row, col)))));
            }
        }
        return m;
    }

    /** Store a word for a specific player (1 or 2)
     * Case insensitive
     * @param player player storing the word
     * @param word word to store
     */
    boolean saveWord(int player, String word) {
        if (word.length() < 3) return false;
        if (player == 1) gameStats.p1Words.add(word.toUpperCase());
        else gameStats.p2Words.add(word.toUpperCase());
        return true;
    }

    void dissaveWord(int player, String word) {
        if (player == 1) gameStats.p1Words.remove(word);
        else gameStats.p2Words.remove(word);
    }

    /** Remove wrong words in players' lists and summarize the current round
     *
     * @return summary info up to now
     */
    public HashMap<Integer, String> summarizeRound() {
        gameStats.p1Words.removeIf(word -> !allWords.containsKey(word));
        gameStats.p2Words.removeIf(word -> !allWords.containsKey(word));
        return gameStats.summarizeRound();
    }


    /** Renew timer of
     *
     * @param minutes Label displaying remaining minutes
     * @param seconds Label displaying remaining seconds
     * @param input text field for player to input words
     */
    public void renewTimer(TextField minutes, TextField seconds, TextField input) {
        if (time == 0) {
            boolean badInput = false; // If the user inputs improper contents in timer
            String min = minutes.getText().strip();
            String sec = seconds.getText().strip();
            int ms = -1;
            int ss = -1;
            if (min.length() == 0) ms = 0;
            else try {
                ms = Integer.parseInt(min);
            } catch (NumberFormatException e) {
                badInput = true;
            }
            if (sec.length() == 0) ss = 0;
            else try {
                ss = Integer.parseInt(sec);
            } catch (NumberFormatException e) {
                badInput = true;
            }
            if (ms == 0 && ss == 0 || ms > 99 || ss > 59 || ms < 0 || ss < 0) badInput = true;
            if (badInput) {
                minutes.setText("");
                seconds.setText("");
                time = -1; // No timer for this round of game
                return;
            }
            time = ms * 60 + ss;
        }
        if (time == -1) return;
        if (timer != null) timer.stop();
        timer = new GameTimer(minutes, seconds, input, time);
    }

    /**
     * Pause when the game is not pause, resume otherwise
     * @return if the game is paused
     */
    public boolean pauseResume() {
        if (timer != null) {
            if  (paused) timer.start();
            else timer.stop();
        }
        paused = !paused;
        return paused;
    }

    /**
     * Refresh Data for a new round
     */
    protected void newRound() {
        gameStats.endRound();
    }


    /**
     * Gets words from the computer.  The computer should find words that are
     * valid。 For each word that the computer finds, update the computer's word
     * list and increment the computer's score (stored in boggleStats).
     *
     */
    public void computerMove(String difficulty){
        if (difficulty.equals("Hard")) {
            CMhard hard = new CMhard(this);
            ArrayList<String> wordchosen = hard.wordchosen;
            hard.chooseRandomWords();
            if (hard.cheating()) {
                this.gameStats.p2Score += 9999999;
            }
            this.gameStats.p2Words.addAll(wordchosen);
        } else if (difficulty.equals("Normal")) {
            CMMedium Medium = new CMMedium(this);
            ArrayList<String> wordchosen = Medium.wordchosen;
            Medium.chooseRandomWords();
            this.gameStats.p2Words.addAll(wordchosen);
        } else {
            CMeasy Easy = new CMeasy(this);
            ArrayList<String> wordchosen = Easy.wordchosen;
            Easy.chooseRandomWords();
            Easy.bug();
            this.gameStats.p2Words.addAll(wordchosen);
        }

    }

    /**
     *
     * @return the grid (y, x)  (0, 0) at the upper left corner
     */
    public char[][] getGrid() {
        return grid.board;
    }

    public Set<String> getComputerWords() {
        return gameStats.p2Words;
    }

    /**
     * Randomly choose and take away/replace 25% of the letters
     */
    public void takeAwayLetters() {
        Random randIndex = new Random();
        int limit = (int) (0.25 * (grid.numRows() * grid.numCols()));

        int i = 0;
        while (i < limit) {
            int chosen_row = randIndex.nextInt(grid.numRows());
            int chosen_col = randIndex.nextInt(grid.numCols());
            while (grid.getCharAt(chosen_row, chosen_col) == ' ') {
                chosen_row = randIndex.nextInt(grid.numRows());
                chosen_col = randIndex.nextInt(grid.numCols());
            }
            grid.replace(chosen_row, chosen_col, ' ');
            i += 1;
        }
        refreshAllWords();
    }

    /**
     * Helper function that helps refresh the allWords attribute after randomly taking away the letters.
     */
    private void refreshAllWords() {
        allWords = new HashMap<>();
        findAllWords();
    }

    /**
     * Test only, get player's word list
     * @param player specification of the player
     * @return the specified player's word list
     */
    public Set<String> getPlayerWords(int player) {
        if (player == 1) return gameStats.p1Words;
        else return gameStats.p2Words;
    }
}
