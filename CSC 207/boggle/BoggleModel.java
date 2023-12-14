package boggle;

/**
 * The specific game, defining the model of the game
 */
public class BoggleModel {
    public String gameMode;
    private int model;

    public BoggleGame game;

    protected BoggleStats stats;

    public String level; // Difficulty of the game: None, Easy, Normal, Hard

    public BoggleModel(String gMode, int size, String difficulty) {
        if (gMode == "PVP") {
            model = 1;
        } else {
            model = 0; // Remember that PVE and PVE: Hole are similar
        }
        gameMode = gMode;
        stats = new BoggleStats();
        game = new BoggleGame(size, stats);
        level = difficulty;
    }

    /**
     * Go into next state of the game (turn, summary, new round)
     * @return return if the game gose into the summary
     */
    public boolean next() {
        if (model == 1) {
            model = 2;
            return false;
        }
        if (model == 0) game.computerMove(this.level);
        return true;
    }

    /**
     * Start a new round
     * @param size Size for the new round
     */
    public void nextRound(int size) {
        if (model == 2) model = 1;
        game = new BoggleGame(size, stats);
        game.newRound();
        if (gameMode == "PVE: HOLE") {
            game.takeAwayLetters();
        }
    }

    /**
     * Add a word for the current player
     * @param word word to be added
     */
    public boolean inputWord(String word) {
        if (model == 2) return game.saveWord(2, word);
        else return game.saveWord(1, word);
    }

    /**
     * Remove a word for the current player
     * @param word word to be removed
     */
    public void removeWord(String word) {
        if (model == 2) game.dissaveWord(2, word);
        else game.dissaveWord(1, word);
    }

    /**
     *
     * @return the difficulty of the game
     */
    public String getLevel() {
        return level;
    }



}
