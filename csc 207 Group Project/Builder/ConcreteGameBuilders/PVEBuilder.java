package Builder.ConcreteGameBuilders;

/**
 * This is a builder with Presets for a PVE gamemode settings.
 */
public class PVEBuilder extends GameBuilder{

    /**
     * Attempts to create a GameBuilder with Preset PVE settings
     */
    public PVEBuilder() {
        super("PVE");
    }

}
