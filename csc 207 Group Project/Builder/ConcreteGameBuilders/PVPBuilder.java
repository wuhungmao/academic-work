package Builder.ConcreteGameBuilders;

/**
 * This is a concrete builder with Preset PVP settings.
 */
public class PVPBuilder extends GameBuilder {

    /**
     * Attempts to create a Game Builder with preset Settings
     */
    public PVPBuilder() {
        super("PVP");
        setDifficulty("None");
    }
}
