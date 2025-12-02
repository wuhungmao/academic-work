package Builder.ConcreteGameBuilders;


import boggle.BoggleModel;

/**
 * Builder for a preset HOLE gamemode
 */
public class HOLEBuilder extends GameBuilder {

    /**
     * Constructor for Hole gamemode
     */
    public HOLEBuilder() {
        super("PVE: HOLE");
    }

    @Override
    public BoggleModel createGame() {
        BoggleModel returned_model = super.createGame();
        returned_model.game.takeAwayLetters();

        return returned_model;
    }
}
