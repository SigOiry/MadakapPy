/*=========================================================================

  Program:   gapfilling
  Language:  C++

  Copyright (c) Jordi Inglada. All rights reserved.

  See LICENCE for details.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _DATEUTILS_H_
#define _DATEUTILS_H_

#include <ctime>
#include <vector>
#include <string>
#include <map>
#include <vnl/vnl_vector.h>
#include <boost/algorithm/string.hpp>

/// Compare 2 dates
bool operator<(std::tm x, std::tm y);

namespace GapFilling {
using PrecisionType = double;
using VectorType = vnl_vector<PrecisionType>;
using DateVector=std::vector<std::tm>;

/// Split a string using a given separator
std::vector<std::string> string_split(const std::string& s, const std::string& sep);

/// return a string with a 0 at the front if the int is < 10
std::string pad_int(int x);

/// Makes a date (std::tm struct) from a string with format YYYYMMDD
std::tm make_date(const std::string& d);

/// Returns the difference in days between 2 time structures
int delta_days(std::tm t1, std::tm t2);

/// Return the day of year
unsigned int doy(const std::tm& d);


/** Functor with state to return the day of the year dealing with
dates spread over multiple years adding 365*year_count to the doy.
Does not take into account leap years.
*/
struct doy_multi_year
{
  doy_multi_year() = default;
  doy_multi_year(unsigned int d_prev_init) : d_prev{d_prev_init}
  {
    first_date = false;
  }
  unsigned int operator()(const std::tm& d)
  {
    auto tmp_doy = doy(d);
    if(!first_date) 
      {
      if(tmp_doy < d_prev-365*year_count)
        ++year_count;
      tmp_doy += 365*year_count;
      }
    else
      { 
      first_date = false;
      }
    d_prev = tmp_doy;
    return tmp_doy;
  };
  unsigned int year_count = 0;
  unsigned int d_prev = 0;
  bool first_date = true;
};

/** Parses an ASCII file containing a date per line (string with format YYYYMMDD)
    and returns an std::vector< std::tm > containing the dates in the order of reading
*/
DateVector parse_date_file(const std::string& df);

/** Parses the first line of a csv file looking for dates like 4-Feb
    and returns an std::vector< std::tm >. The year is passed as parameter. Returns a
    pair containing a vector of dates and the starting position (column) of the dates
    in the file.
*/
std::pair<DateVector, int> parse_dates_csv_file(const std::string& df, int year);

/** Extracts the profiles from a csv file (MAISEO format). Returns a vector of profiles.
    Gets as parameters the file name, the starting position of the dates (column) and the
    number of availble dates in the file.**/
std::vector<VectorType> get_csv_profiles(const std::string& fname, int pos, int ndates);

/** Parses the first line of a cvs file with this format:
    parcelle;f090215;f090317;f090321;f090603;f090623;f090701;f090712;f090726;f090805;f090814;f090830;f090906;f090924;f090930;f091016;f091225
    and returns a DateVector
*/
DateVector parse_dates_sirhyus(const std::string& fname);

/** Extracts the profiles from a csv file (Sirhyus format). Returns a vector of pairs
    where each pair is <field id, profile>
    Gets as parameters the file name.**/
std::vector<std::pair<long int, VectorType>> get_sirhyus_profiles(const std::string& df, 
                                                                  unsigned int nbDates);
/// Map to convert months from text to int
int months(const std::string& m);
}

#endif
