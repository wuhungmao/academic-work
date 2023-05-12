package Builder;

import boggle.BoggleModel;


/**
 * AbstractGUIGameBuilder Class to inherit for future GameMode + setting creations on GUI!
 */
public abstract class AbstractGUIGameBuilder implements GameModeBuilderInterface {

    protected String gameMode;
    protected String difficulty;
    protected int size;

    /**
     * Set the Game Difficulty settings
     */
    public abstract void setDifficulty(String diff);

    /**
     * Set the size settings
     */
    public abstract void setSize(int sizeOfGrid);

    /**
     * Sub-Classes of this Abstract, MUST implement this createGame()
     *
     * @return BoggleModel of desired gameMode and settings
     */
    public abstract BoggleModel createGame();
}
