package Builder;


import boggle.BoggleModel;

/**
 * Interface more making GameModes
 */
public interface GameModeBuilderInterface {

    /**
     * Create the Game
     *
     * @return BoggleModel that reflects the Game MODE and settings of the user's desire.
     */
    BoggleModel createGame();
}
