#ifndef ATOM_TAG_PARSERS_H_
#define ATOM_TAG_PARSERS_H_
#include <string>
#include <vector>

class Atom_Tag_Record {
 public:
  Atom_Tag_Record(std::string, std::string, int);
  
  bool compare_name(const std::string&);
  bool compare_symbol(const std::string&);
  int symbol_length();
  int get_charge();

  std::string symbol();

 private:
  std::string atom_name;
  std::string atom_symbol;
  int charge;
};

class Atom_Tag_Parser {
 public:
  Atom_Tag_Parser() {
    atom_tag_records.emplace_back("hydrogen"     , "h" , 1  );
    atom_tag_records.emplace_back("helium"       , "he", 2  );
    atom_tag_records.emplace_back("lithium"      , "li", 3  );
    atom_tag_records.emplace_back("beryllium"    , "be", 4  );
    atom_tag_records.emplace_back("boron"        , "b" , 5  );
    atom_tag_records.emplace_back("carbon"       , "c" , 6  );
    atom_tag_records.emplace_back("nitrogen"     , "n" , 7  );
    atom_tag_records.emplace_back("oxygen"       , "o" , 8  );
    atom_tag_records.emplace_back("fluorine"     , "f" , 9  );
    atom_tag_records.emplace_back("neon"         , "ne", 10 );
    atom_tag_records.emplace_back("sodium"       , "na", 11 );
    atom_tag_records.emplace_back("magnesium"    , "mg", 12 );
    atom_tag_records.emplace_back("aluminium"    , "al", 13 );
    atom_tag_records.emplace_back("silicon"      , "si", 14 );
    atom_tag_records.emplace_back("phosphorus"   , "p" , 15 );
    atom_tag_records.emplace_back("sulfur"       , "s" , 16 );
    atom_tag_records.emplace_back("chlorine"     , "cl", 17 );
    atom_tag_records.emplace_back("argon"        , "ar", 18 );
    atom_tag_records.emplace_back("potassium"    , "k" , 19 );
    atom_tag_records.emplace_back("calcium"      , "ca", 20 );
    atom_tag_records.emplace_back("scandium"     , "sc", 21 );
    atom_tag_records.emplace_back("titanium"     , "ti", 22 );
    atom_tag_records.emplace_back("vanadium"     , "v" , 23 );
    atom_tag_records.emplace_back("chromium"     , "cr", 24 );
    atom_tag_records.emplace_back("manganese"    , "mn", 25 );
    atom_tag_records.emplace_back("iron"         , "fe", 26 );
    atom_tag_records.emplace_back("cobalt"       , "co", 27 );
    atom_tag_records.emplace_back("nickel"       , "ni", 28 );
    atom_tag_records.emplace_back("copper"       , "cu", 29 );
    atom_tag_records.emplace_back("zinc"         , "zn", 30 );
    atom_tag_records.emplace_back("gallium"      , "ga", 31 );
    atom_tag_records.emplace_back("germanium"    , "ge", 32 );
    atom_tag_records.emplace_back("arsenic"      , "as", 33 );
    atom_tag_records.emplace_back("selenium"     , "se", 34 );
    atom_tag_records.emplace_back("bromine"      , "br", 35 );
    atom_tag_records.emplace_back("krypton"      , "kr", 36 );
    atom_tag_records.emplace_back("rubidium"     , "rb", 37 );
    atom_tag_records.emplace_back("strontium"    , "sr", 38 );
    atom_tag_records.emplace_back("yttrium"      , "y" , 39 );
    atom_tag_records.emplace_back("zirconium"    , "zr", 40 );
    atom_tag_records.emplace_back("niobium"      , "nb", 41 );
    atom_tag_records.emplace_back("molybdenum"   , "mo", 42 );
    atom_tag_records.emplace_back("technetium"   , "tc", 43 );
    atom_tag_records.emplace_back("ruthenium"    , "ru", 44 );
    atom_tag_records.emplace_back("rhodium"      , "rh", 45 );
    atom_tag_records.emplace_back("palladium"    , "pd", 46 );
    atom_tag_records.emplace_back("silver"       , "ag", 47 );
    atom_tag_records.emplace_back("cadmium"      , "cd", 48 );
    atom_tag_records.emplace_back("indium"       , "in", 49 );
    atom_tag_records.emplace_back("tin"          , "sn", 50 );
    atom_tag_records.emplace_back("antimony"     , "sb", 51 );
    atom_tag_records.emplace_back("tellurium"    , "te", 52 );
    atom_tag_records.emplace_back("iodine"       , "i" , 53 );
    atom_tag_records.emplace_back("xenon"        , "xe", 54 );
    atom_tag_records.emplace_back("caesium"      , "cs", 55 );
    atom_tag_records.emplace_back("barium"       , "ba", 56 );
    atom_tag_records.emplace_back("lanthanum"    , "la", 57 );
    atom_tag_records.emplace_back("cerium"       , "ce", 58 );
    atom_tag_records.emplace_back("praseodymium" , "pr", 59 );
    atom_tag_records.emplace_back("neodymium"    , "nd", 60 );
    atom_tag_records.emplace_back("promethium"   , "pm", 61 );
    atom_tag_records.emplace_back("samarium"     , "sm", 62 );
    atom_tag_records.emplace_back("europium"     , "eu", 63 );
    atom_tag_records.emplace_back("gadolinium"   , "gd", 64 );
    atom_tag_records.emplace_back("terbium"      , "tb", 65 );
    atom_tag_records.emplace_back("dysprosium"   , "dy", 66 );
    atom_tag_records.emplace_back("holmium"      , "ho", 67 );
    atom_tag_records.emplace_back("erbium"       , "er", 68 );
    atom_tag_records.emplace_back("thulium"      , "tm", 69 );
    atom_tag_records.emplace_back("ytterbium"    , "yb", 70 );
    atom_tag_records.emplace_back("lutetium"     , "lu", 71 );
    atom_tag_records.emplace_back("hafnium"      , "hf", 72 );
    atom_tag_records.emplace_back("tantalum"     , "ta", 73 );
    atom_tag_records.emplace_back("tungsten"     , "w" , 74 );
    atom_tag_records.emplace_back("rhenium"      , "re", 75 );
    atom_tag_records.emplace_back("osmium"       , "os", 76 );
    atom_tag_records.emplace_back("iridium"      , "ir", 77 );
    atom_tag_records.emplace_back("platinum"     , "pt", 78 );
    atom_tag_records.emplace_back("gold"         , "au", 79 );
    atom_tag_records.emplace_back("mercury"      , "hg", 80 );
    atom_tag_records.emplace_back("thallium"     , "tl", 81 );
    atom_tag_records.emplace_back("lead"         , "pb", 82 );
    atom_tag_records.emplace_back("bismuth"      , "bi", 83 );
    atom_tag_records.emplace_back("polonium"     , "po", 84 );
    atom_tag_records.emplace_back("astatine"     , "at", 85 );
    atom_tag_records.emplace_back("radon"        , "rn", 86 );
    atom_tag_records.emplace_back("francium"     , "fr", 87 );
    atom_tag_records.emplace_back("radium"       , "ra", 88 );
    atom_tag_records.emplace_back("actinium"     , "ac", 89 );
    atom_tag_records.emplace_back("thorium"      , "th", 90 );
    atom_tag_records.emplace_back("protactinium" , "pa", 91 );
    atom_tag_records.emplace_back("uranium"      , "u" , 92 );
    atom_tag_records.emplace_back("neptunium"    , "np", 93 );
    atom_tag_records.emplace_back("plutonium"    , "pu", 94 );
    atom_tag_records.emplace_back("americium"    , "am", 95 );
    atom_tag_records.emplace_back("curium"       , "cm", 96 );
    atom_tag_records.emplace_back("berkelium"    , "bk", 97 );
    atom_tag_records.emplace_back("californium"  , "cf", 98 );
    atom_tag_records.emplace_back("einsteinium"  , "es", 99 );
    atom_tag_records.emplace_back("fermium"      , "fm", 100);
    atom_tag_records.emplace_back("mendelevium"  , "md", 101);
    atom_tag_records.emplace_back("nobelium"     , "no", 102);
    atom_tag_records.emplace_back("lawrencium"   , "lr", 103);
    atom_tag_records.emplace_back("rutherfordium", "rf", 104);
    atom_tag_records.emplace_back("dubnium"      , "db", 105);
    atom_tag_records.emplace_back("seaborgium"   , "sg", 106);
    atom_tag_records.emplace_back("bohrium"      , "bh", 107);
    atom_tag_records.emplace_back("hassium"      , "hs", 108);
    atom_tag_records.emplace_back("meitnerium"   , "mt", 109);
    atom_tag_records.emplace_back("darmstadtium" , "ds", 110);
    atom_tag_records.emplace_back("roentgenium"  , "rg", 111);
    atom_tag_records.emplace_back("copernicium"  , "cn", 112);
    atom_tag_records.emplace_back("nihonium"     , "nh", 113);
    atom_tag_records.emplace_back("flerovium"    , "fl", 114);
    atom_tag_records.emplace_back("moscovium"    , "mc", 115);
    atom_tag_records.emplace_back("livermorium"  , "lv", 116);
    atom_tag_records.emplace_back("tennessine"   , "ts", 117);
    atom_tag_records.emplace_back("oganesson"    , "og", 118);
  }
  int parse(std::string);
  std::string symbol(int);
 private:
  std::vector<Atom_Tag_Record> atom_tag_records;
};
#endif  // ATOM_TAG_PARSERS_H_
