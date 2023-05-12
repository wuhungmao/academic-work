package Builder.ConcreteGameBuilders;

import Builder.AbstractGUIGameBuilder;
import boggle.BoggleModel;
import javafx.scene.control.TextField;


/**
 * This is a very General Concrete Builder that builds based on gamemode and difficulty
 */
public class GameBuilder extends AbstractGUIGameBuilder {
    protected String gameMode;
    protected String difficulty = "Easy";
    protected int size = 4;

    /**
     * initialize the GameModeBuilder for future settings usage
     */
    public GameBuilder(String gMode) { gameMode = gMode; };

    @Override
    public void setDifficulty(String diff) {
        difficulty = diff;
    }

    @Override
    public void setSize(int sizeOfGrid) {
        size = sizeOfGrid;
    }

    @Override
    public BoggleModel createGame() {
        return new BoggleModel(gameMode, size, difficulty);
    }


}
