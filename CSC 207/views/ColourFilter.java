package views;

import javafx.scene.paint.Color;

import java.awt.*;
import java.util.HashMap;
import java.util.Map;

public class ColourFilter {
    /**
     * The three primary colours, useful for instantiation.
     */
    public static final String[] PRIMARY_COLOURS = new String[]{
            "RED", "GREEN", "BLUE"
    };

    /**
     * The minimum values for the three primary colours.
     */
    private final Map<String, Double> infima;

    /**
     * The maximum values for the three primary colours.
     */
    private final Map<String, Double> suprema;

    /**
     * The constructor.
     * @param minima the minimum values.
     * @param maxima the maximum values.
     */
    public ColourFilter(final Map<String, Double> minima, final Map<String, Double> maxima) {
        this.infima = new HashMap<>();
        this.initialise_i(minima);
        this.suprema = new HashMap<>();
        this.initialise_s(maxima);
    }

    /**
     * Filter a given colour.
     * @param colour
     * @return true if the colour passes the filter, false otherwise.
     */
    public boolean filter(final Color colour) {
        double red = colour.getRed();
        double green = colour.getGreen();
        double blue = colour.getBlue();

        return red >= this.infima.get("RED") && red <= this.suprema.get("RED")
                && green >= this.infima.get("GREEN") && green <= this.suprema.get("GREEN")
                && blue >= this.infima.get("BLUE") && blue <= this.suprema.get("BLUE");
    }

    /**
     * Calculate how much possible RGB colours is kept by filter. A return value of 0.5 implies that 50% of the colours
     * passes this filter.
     * @return the image of the codomain.
     */
    public double image() {
        return (this.suprema.get("RED") - this.infima.get("RED"))
                * (this.suprema.get("GREEN") - this.infima.get("GREEN"))
                * (this.suprema.get("BLUE") - this.infima.get("BLUE"));
    }

    /**
     * Initialise the minimum values. The default minimum value is 0.
     * @param minima
     */
    private void initialise_i(final Map<String, Double> minima) {
        for (final String primary: ColourFilter.PRIMARY_COLOURS) {
            if (minima.containsKey(primary)) {
                double value = minima.get(primary);
                value = ColourFilter.normalise_i(value);
                this.infima.put(primary, value);
            } else {
                this.infima.put(primary, 0.);
            }
        }
    }

    /**
     * Initialise the maximum values. The default maximum value is 1.
     * @param maxima
     */
    private void initialise_s(final Map<String, Double> maxima) {
        for (final String primary: ColourFilter.PRIMARY_COLOURS) {
            if (maxima.containsKey(primary)) {
                double value = maxima.get(primary);
                value = ColourFilter.normalise_s(value);
                this.suprema.put(primary, value);
            } else {
                this.suprema.put(primary, 1.);
            }
        }
    }

    /**
     * Normalise a value into a minimum value in the interval [0, 1]. The default value 0 is used for Infinite and NaN.
     * @param value
     * @return
     */
    private static double normalise_i(final double value) {
        double orthonormal = value;
        if (Double.isInfinite(value) || Double.isNaN(value)) {
            return 0.;
        }
        orthonormal = orthonormal < 0. ? 0. : orthonormal;
        orthonormal = orthonormal > 1. ? 1. : orthonormal;
        return orthonormal;
    }

    /**
     * Normalise a value into a maximum value in the interval [0, 1]. The default value 1 is used for Infinite and NaN.
     * @param value
     * @return
     */
    private static double normalise_s(final double value) {
        double orthonormal = value;
        if (Double.isInfinite((value)) || Double.isNaN(value)) {
            return 1.;
        }
        orthonormal = orthonormal > 1. ? 1. : orthonormal;
        orthonormal = orthonormal < 0. ? 0. : orthonormal;
        return orthonormal;
    }
}
