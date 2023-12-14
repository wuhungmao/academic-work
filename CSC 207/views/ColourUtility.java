package views;

import javafx.scene.paint.Color;

import java.util.*;
import java.util.List;

/**
 * The ColourUtility class provides a helper method getColours() that generates a List of a number of colours that are
 * distinct enough under a provided colourblindness setting.
 */
public class ColourUtility {
    /**
     * The minimum difference seen by a person adjusted by the number of colours it sees. For example, for a
     * trichromatic (normal) person, the value is adjusted by a multiplier of 3; for a monochromatic (total colourblind)
     * person, the value is adjusted by a multiplier of 1.
     */
    private static final double MINIMUM_DIFFERENCE = 0.05;

    /**
     * The colourblindness parameter used in the helper getColours() method.
     */
    public enum Colourblindness{
        NORMAL,
        RED_GREEN,
        BLUE_YELLOW,
        TOTAL
    }

    /**
     * The default pseudorandom number generator in this class.
     */
    private static final Random RANDOM = new Random();

    /**
     * The private constructor for a helper class.
     */
    private ColourUtility() {
    }

    /**
     * Produces an ArrayList containing colours that are distinct enough under the provided colourblind settings.
     * @param size the size of the ArrayList of colours, ranging from 0 - 10 (inclusive). An empty list is returned if the
     *             parameter is less than or equal to 0.
     * @param colourblindness colourblindness settings.
     * @return An ArrayList with size number of Color elements to be used.
     * @throws IllegalArgumentException Exception is thrown if parameter size is greater than 10. The constants are
     * adjusted so that the best results cannot be produced.
     */
    public static List<Color> getColourList(final int size, final Colourblindness colourblindness)
            throws IllegalArgumentException{
        return ColourUtility.getColourList(size, colourblindness, ColourUtility.getDefaultFilter());
    }


    /**
     * The iterator version of getColourList().
     * @param size
     * @param colourblindness
     * @return
     */
    public static Iterator<Color> getColourIterator(final int size, final Colourblindness colourblindness) {
        return new ColourIterator(ColourUtility.getColourList(size, colourblindness));
    }

    /**
     * Produces an Iterator containing colours that are distinct enough under the provided colourblind settings.
     * @param size the size of the ArrayList of colours, ranging from 0 - (10 * filtering ratio) (inclusive). An empty
     *             list is returned if the parameter is less than or equal to 0.
     * @param colourblindness colourblindness settings.
     * @return An Iterator with size number of Color elements to be used.
     */
    public static List<Color> getColourList(final int size, final Colourblindness colourblindness,
                                            final ColourFilter filter) throws IllegalArgumentException{
        if (size > filter.image() * 10) {
            throw new IllegalArgumentException("Invalid parameter presents: size. The maximum supported is 10 " +
                    "multiplied by the preservation ratio of the filter.");
        }

        ArrayList<Color> colours = new ArrayList<>();

        if (size <= 0) {
            return colours;
        }

        OUTER: while (true) {
            // select a random standard JavaFX colour
            Color random = ColourUtility.COLOURS[ColourUtility.RANDOM.nextInt(ColourUtility.COLOURS.length)];

            if (!filter.filter(random)) {
                continue;
            }

            // compare to each already selected colours
            for (Color colour: colours) {
                switch (colourblindness) {
                    // in the default case, all primary colours are considered contributing to the difference
                    case NORMAL:
                        if (Math.abs(random.getRed() - colour.getRed()) + Math.abs(random.getGreen()
                                - colour.getGreen()) + Math.abs(random.getBlue() - colour.getBlue())
                                < 3 * MINIMUM_DIFFERENCE) {
                            continue OUTER;
                        }
                        break;

                    // in the red-green colourblind case, red and green are seen indifferently
                    case RED_GREEN:
                        if (Math.abs(random.getRed() + random.getGreen() - colour.getRed() - colour.getGreen())
                                + Math.abs(random.getBlue() - colour.getBlue()) < 2 * MINIMUM_DIFFERENCE) {
                            continue OUTER;
                        }
                        break;

                    // in the blue-yellow colourblind case, green and blue are seen indifferently
                    case BLUE_YELLOW:
                        if (Math.abs(random.getRed() - colour.getRed()) + Math.abs(random.getGreen() + colour.getGreen()
                                - random.getBlue() - colour.getBlue()) < 2 * MINIMUM_DIFFERENCE) {
                            continue OUTER;
                        }
                        break;

                    // in the total colourblind case, only greyness matters to select colours
                    case TOTAL:
                        if (Math.abs(random.getRed() + random.getGreen() + random.getBlue() - colour.getRed()
                                - colour.getGreen() - colour.getBlue()) < MINIMUM_DIFFERENCE) {
                            continue OUTER;
                        }
                        break;

                    default:
                        break;
                }
            }

            colours.add(random);

            // if enough number of colour is generated
            if (colours.size() >= size) {
                return colours;
            }
        }
    }

    /**
     * The iterator version of getColourList().
     * @param size
     * @param colourblindness
     * @return
     */
    public static Iterator<Color> getColourIterator(final int size, final Colourblindness colourblindness,
                                                    final ColourFilter filter) {
        return new ColourIterator(ColourUtility.getColourList(size, colourblindness, filter));
    }


    public static ColourFilter getDefaultFilter() {
        Map<String, Double> minima = new HashMap<>();
        minima.put("RED", 0.);
        minima.put("GREEN", 0.);
        minima.put("BLUE", 0.);

        Map<String, Double> maxima = new HashMap<>();
        maxima.put("RED", 1.);
        maxima.put("GREEN", 1.);
        maxima.put("BLUE", 1.);

        ColourFilter filter = new ColourFilter(minima, maxima);
        return filter;
    }


    /**
     * Filter a list of colours with a provided filter.
     * @param colours the List of Color elements to be filters.
     * @param filter the ColorFilter to use.
     * @return a filtered list.
     */
    public static List<Color> filterColourList(final List<Color> colours, final ColourFilter filter) {
        List<Color> filtered = new ArrayList<>();

        for (Color colour: colours) {
            if (filter.filter(colour)) {
                filtered.add(colour);
            }
        }

        return filtered;
    }

    /**
     * Formal names for colourblindness, useful for the user. Note that both Protanopia and Deuteranopia result in the
     * Red-Green Colourblindness.
     */
    public static final String[] OPHTHALMOLOGICAL_ARGUMENT = new String[] {
            "DEFAULT", "PROTANOPIA", "DEUTERANOPIA", "TRITANOPIA", "ACHROMATOPSIA"
    };

    /**
     * The standard colours by JavaFX.
     */
    public static final Color[] COLOURS = new Color[]{
            //Color.TRANSPARENT,
            Color.ALICEBLUE,
            Color.ANTIQUEWHITE,
            Color.AQUA,
            Color.AQUAMARINE,
            Color.AZURE,
            Color.BEIGE,
            Color.BISQUE,
            Color.BLACK,
            Color.BLANCHEDALMOND,
            Color.BLUE,
            Color.BLUEVIOLET,
            Color.BROWN,
            Color.BURLYWOOD,
            Color.CADETBLUE,
            Color.CHARTREUSE,
            Color.CHOCOLATE,
            Color.CORAL,
            Color.CORNFLOWERBLUE,
            Color.CORNSILK,
            Color.CRIMSON,
            Color.CYAN,
            Color.DARKBLUE,
            Color.DARKCYAN,
            Color.DARKGOLDENROD,
            Color.DARKGRAY,
            Color.DARKGREEN,
            Color.DARKGREY,
            Color.DARKKHAKI,
            Color.DARKMAGENTA,
            Color.DARKOLIVEGREEN,
            Color.DARKORANGE,
            Color.DARKORCHID,
            Color.DARKRED,
            Color.DARKSALMON,
            Color.DARKSEAGREEN,
            Color.DARKSLATEBLUE,
            Color.DARKSLATEGRAY,
            Color.DARKSLATEGREY,
            Color.DARKTURQUOISE,
            Color.DARKVIOLET,
            Color.DEEPPINK,
            Color.DEEPSKYBLUE,
            Color.DIMGRAY,
            Color.DIMGREY,
            Color.DODGERBLUE,
            Color.FIREBRICK,
            Color.FLORALWHITE,
            Color.FORESTGREEN,
            Color.FUCHSIA,
            Color.GAINSBORO,
            Color.GHOSTWHITE,
            Color.GOLD,
            Color.GOLDENROD,
            Color.GRAY,
            Color.GREEN,
            Color.GREENYELLOW,
            Color.GREY,
            Color.HONEYDEW,
            Color.HOTPINK,
            Color.INDIANRED,
            Color.INDIGO,
            Color.IVORY,
            Color.KHAKI,
            Color.LAVENDER,
            Color.LAVENDERBLUSH,
            Color.LAWNGREEN,
            Color.LEMONCHIFFON,
            Color.LIGHTBLUE,
            Color.LIGHTCORAL,
            Color.LIGHTCYAN,
            Color.LIGHTGOLDENRODYELLOW,
            Color.LIGHTGRAY,
            Color.LIGHTGREEN,
            Color.LIGHTGREY,
            Color.LIGHTPINK,
            Color.LIGHTSALMON,
            Color.LIGHTSEAGREEN,
            Color.LIGHTSKYBLUE,
            Color.LIGHTSLATEGRAY,
            Color.LIGHTSLATEGREY,
            Color.LIGHTSTEELBLUE,
            Color.LIGHTYELLOW,
            Color.LIME,
            Color.LIMEGREEN,
            Color.LINEN,
            Color.MAGENTA,
            Color.MAROON,
            Color.MEDIUMAQUAMARINE,
            Color.MEDIUMBLUE,
            Color.MEDIUMORCHID,
            Color.MEDIUMPURPLE,
            Color.MEDIUMSEAGREEN,
            Color.MEDIUMSLATEBLUE,
            Color.MEDIUMSPRINGGREEN,
            Color.MEDIUMTURQUOISE,
            Color.MEDIUMVIOLETRED,
            Color.MIDNIGHTBLUE,
            Color.MINTCREAM,
            Color.MISTYROSE,
            Color.MOCCASIN,
            Color.NAVAJOWHITE,
            Color.NAVY,
            Color.OLDLACE,
            Color.OLIVE,
            Color.OLIVEDRAB,
            Color.ORANGE,
            Color.ORANGERED,
            Color.ORCHID,
            Color.PALEGOLDENROD,
            Color.PALEGREEN,
            Color.PALETURQUOISE,
            Color.PALEVIOLETRED,
            Color.PAPAYAWHIP,
            Color.PEACHPUFF,
            Color.PERU,
            Color.PINK,
            Color.PLUM,
            Color.POWDERBLUE,
            Color.PURPLE,
            Color.RED,
            Color.ROSYBROWN,
            Color.ROYALBLUE,
            Color.SADDLEBROWN,
            Color.SALMON,
            Color.SANDYBROWN,
            Color.SEAGREEN,
            Color.SEASHELL,
            Color.SIENNA,
            Color.SILVER,
            Color.SKYBLUE,
            Color.SLATEBLUE,
            Color.SLATEGRAY,
            Color.SLATEGREY,
            Color.SNOW,
            Color.SPRINGGREEN,
            Color.STEELBLUE,
            Color.TAN,
            Color.TEAL,
            Color.THISTLE,
            Color.TOMATO,
            Color.TURQUOISE,
            Color.VIOLET,
            Color.WHEAT,
            Color.WHITE,
            Color.WHITESMOKE,
            Color.YELLOW,
            Color.YELLOWGREEN,
    };
}
