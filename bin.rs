// #![forbid(unused)]
//! Derived from:
//! <https://raw.githubusercontent.com/quickfur/dcal/master/dcal.d>.

/// Date representation.
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct NaiveDate(i32, u32, u32);

impl NaiveDate {
    pub fn from_ymd(y: i32, m: u32, d: u32) -> NaiveDate {
        assert!(1 <= m && m <= 12, "m = {:?}", m);
        assert!(1 <= d && d <= NaiveDate(y, m, 1).days_in_month(), "d = {:?}", d);
        NaiveDate(y, m, d)
    }

    pub fn year(&self) -> i32 {
        self.0
    }

    pub fn month(&self) -> u32 {
        self.1
    }

    pub fn day(&self) -> u32 {
        self.2
    }

    pub fn succ(&self) -> NaiveDate {
        let (mut y, mut m, mut d, n) = (
            self.year(), self.month(), self.day()+1, self.days_in_month());
        if d > n {
            d = 1;
            m += 1;
        }
        if m > 12 {
            m = 1;
            y += 1;
        }
        NaiveDate::from_ymd(y, m, d)
    }

    pub fn weekday(&self) -> Weekday {
        use Weekday::*;

        // 0 = Sunday
        let year = self.year();
        let dow_jan_1 = (year*365 + ((year-1) / 4) - ((year-1) / 100) + ((year-1) / 400)) % 7;
        let dow = (dow_jan_1 + (self.day_of_year() as i32 - 1)) % 7;
        [Sun, Mon, Tue, Wed, Thu, Fri, Sat][dow as usize]
    }

    pub fn isoweekdate(&self) -> (i32, u32, Weekday) {
        let first_dow_mon_0 = self.year_first_day_of_week().num_days_from_monday();

        // Work out this date's DOtY and week number, not including year adjustment.
        let doy_0 = self.day_of_year() - 1;
        let mut week_mon_0: i32 = ((first_dow_mon_0 + doy_0) / 7) as i32;

        if self.first_week_in_prev_year() {
            week_mon_0 -= 1;
        }

        let weeks_in_year = self.last_week_number();

        // Work out the final result.  If the week is -1 or >= weeks_in_year, we will need to adjust the year.
        let year = self.year();
        let wd = self.weekday();

        if week_mon_0 < 0 {
            (year - 1, NaiveDate::from_ymd(year - 1, 1, 1).last_week_number(), wd)
        } else if week_mon_0 >= weeks_in_year as i32 {
            (year + 1, (week_mon_0 + 1 - weeks_in_year as i32) as u32, wd)
        } else {
            (year, (week_mon_0 + 1) as u32, wd)
        }
    }

    fn first_week_in_prev_year(&self) -> bool {
        let first_dow_mon_0 = self.year_first_day_of_week().num_days_from_monday();

        // Any day in the year *before* the first Monday of that year is considered to be in the last week of the previous year, assuming the first week has *less* than four days in it.  Adjust the week appropriately.
        ((7 - first_dow_mon_0) % 7) < 4
    }

    fn year_first_day_of_week(&self) -> Weekday {
        NaiveDate::from_ymd(self.year(), 1, 1).weekday()
    }

    fn weeks_in_year(&self) -> u32 {
        let days_in_last_week = self.year_first_day_of_week().num_days_from_monday() + 1;
        if days_in_last_week >= 4 { 53 } else { 52 }
    }

    fn last_week_number(&self) -> u32 {
        let wiy = self.weeks_in_year();
        if self.first_week_in_prev_year() { wiy - 1 } else { wiy }
    }

    fn day_of_year(&self) -> u32 {
        (1..self.1).map(|m| NaiveDate::from_ymd(self.year(), m, 1).days_in_month())
            .fold(0, |a,b| a+b) + self.day()
    }

    fn is_leap_year(&self) -> bool {
        let year = self.year();
        if year % 4 != 0 {
            return false
        } else if year % 100 != 0 {
            return true
        } else if year % 400 != 0 {
            return false
        } else {
            return true
        }
    }

    fn days_in_month(&self) -> u32 {
        match self.month() {
            /* Jan */ 1 => 31,
            /* Feb */ 2 => if self.is_leap_year() { 29 } else { 28 },
            /* Mar */ 3 => 31,
            /* Apr */ 4 => 30,
            /* May */ 5 => 31,
            /* Jun */ 6 => 30,
            /* Jul */ 7 => 31,
            /* Aug */ 8 => 31,
            /* Sep */ 9 => 30,
            /* Oct */ 10 => 31,
            /* Nov */ 11 => 30,
            /* Dec */ 12 => 31,
            _ => unreachable!()
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Weekday {
    Mon,
    Tue,
    Wed,
    Thu,
    Fri,
    Sat,
    Sun,
}

impl Weekday {
    pub fn num_days_from_monday(&self) -> u32 {
        use Weekday::*;
        match *self {
            Mon => 0,
            Tue => 1,
            Wed => 2,
            Thu => 3,
            Fri => 4,
            Sat => 5,
            Sun => 6,
        }
    }

    pub fn num_days_from_sunday(&self) -> u32 {
        use Weekday::*;
        match *self {
            Sun => 0,
            Mon => 1,
            Tue => 2,
            Wed => 3,
            Thu => 4,
            Fri => 5,
            Sat => 6,
        }
    }
}

/// GroupBy implementation.
struct GroupBy<G, It, F>
where It: Iterator,
    F: FnMut(&It::Item) -> G,
    G: Eq,
{
    it: It,
    g: F,
    next: Option<It::Item>,
}

impl<G, It, F> Iterator for GroupBy<G, It, F>
where It: Iterator,
    F: FnMut(&It::Item) -> G,
    G: Eq
{
    type Item = (G, Vec<It::Item>);

    fn next(&mut self) -> Option<Self::Item> {
        match self.next.take().or_else(|| self.it.next()) {
            None => None,
            Some(e) => {
                let key = (self.g)(&e);
                let mut vs = vec![e];
                'fill_group:
                for e in self.it.by_ref() {
                    match key == (self.g)(&e) {
                        true => vs.push(e),
                        false => {
                            self.next = Some(e);
                            break 'fill_group;
                        }
                    }
                }
                Some((key, vs))
            }
        }
    }
}

trait IteratorExt: Iterator + Sized {
    fn foreach<F>(&mut self, mut f: F)
    where F: FnMut(Self::Item) {
        for e in self { f(e) }
    }

    fn group_by<G, F>(self, g: F) -> GroupBy<G, Self, F>
    where F: FnMut(&Self::Item) -> G,
        G: Eq
    {
        GroupBy {
            it: self,
            g: g,
            next: None,
        }
    }

    fn join(mut self, sep: &str) -> String
    where Self::Item: std::fmt::Display {
        let mut s = String::new();
        if let Some(e) = self.next() {
            s.push_str(&format!("{}", e));
            for e in self {
                s.push_str(sep);
                s.push_str(&format!("{}", e));
            }
        }
        s
    }
}

impl<It> IteratorExt for It where It: Iterator {}

///
/// Generates an iterator that yields exactly n spaces.
///
fn spaces(n: usize) -> std::iter::Take<std::iter::Repeat<char>> {
    std::iter::repeat(' ').take(n)
}

fn test_spaces() {
    assert_eq!(spaces(0).collect::<String>(), "");
    assert_eq!(spaces(10).collect::<String>(), "          ")
}

///
/// Returns an iterator of dates in a given year.
///

// NOTE: In the spirit of the D example, we generate a lazy sequence.  Sadly, since Rust lacks return type deduction, this is rather more verbose that it might otherwise need to be.  An additional pain is that closure types are unnamed, so we can't use those *either*.

fn dates_in_year(year: i32) -> DatesInYear {
    DatesInYear {
        next: NaiveDate::from_ymd(year, 1, 1),
        in_year: year,
    }
}

struct DatesInYear {
    next: NaiveDate,
    in_year: i32,
}

impl Iterator for DatesInYear {
    type Item = NaiveDate;

    fn next(&mut self) -> Option<NaiveDate> {
        if self.next.year() > self.in_year {
            None
        } else {
            let item = self.next;
            self.next = self.next.succ();
            Some(item)
        }
    }
}

fn test_dates_in_year() {
    {
        let mut dates = dates_in_year(2013);
        assert_eq!(dates.next(), Some(NaiveDate::from_ymd(2013, 1, 1)));

        // Check increment
        assert_eq!(dates.next(), Some(NaiveDate::from_ymd(2013, 1, 2)));

        // Check monthly rollover
        for _ in 3..31 {
            assert!(dates.next() != None);
        }

        assert_eq!(dates.next(), Some(NaiveDate::from_ymd(2013, 1, 31)));
        assert_eq!(dates.next(), Some(NaiveDate::from_ymd(2013, 2, 1)));
    }

    {
        // Check length of year
        let mut dates = dates_in_year(2013);
        for _ in 0..365 {
            assert!(dates.next() != None);
        }
        assert_eq!(dates.next(), None);
    }

    {
        // Check length of leap year
        let mut dates = dates_in_year(1984);
        for _ in 0..366 {
            assert!(dates.next() != None);
        }
        assert_eq!(dates.next(), None);
    }
}

///
/// Convenience trait for verifying that a given type iterates over
/// `NaiveDate`s.
///
trait DateIterator: Iterator<Item=NaiveDate> {}
impl<It> DateIterator for It where It: Iterator<Item=NaiveDate> {}

/*
I am not doing an implementation of `chunkBy`.  Instead, I'll just use
`itertools::Itertools::group_by`.

This is one place where Rust really bites us: the result of the D `chunkBy`
is a lazy sequence of lazy sequences.  That is quite hard to do in Rust without
a *mountain* of code.  Part of the problem is the implicit sharing that has to
happen for it to work.

To give you an idea of how involved it can be, see:
<https://github.com/DanielKeep/rust-grabbag/blob/master/src/iter/group_by.rs>.
*/

fn test_group_by() {
    let input = [
        [1, 1],
        [1, 1],
        [1, 2],
        [2, 2],
        [2, 3],
        [2, 3],
        [3, 3]
    ];

    let by_x = input.iter().cloned().group_by(|a| a[0]);
    let expected_1: &[&[[i32; 2]]] = &[
        &[[1, 1], [1, 1], [1, 2]],
        &[[2, 2], [2, 3], [2, 3]],
        &[[3, 3]]
    ];
    for ((_, a), b) in by_x.zip(expected_1.iter().cloned()) {
        assert_eq!(&a[..], b);
    }

    let by_y = input.iter().cloned().group_by(|a| a[1]);
    let expected_2: &[&[[i32; 2]]] = &[
        &[[1, 1], [1, 1]],
        &[[1, 2], [2, 2]],
        &[[2, 3], [2, 3], [3, 3]]
    ];
    for ((_, a), b) in by_y.zip(expected_2.iter().cloned()) {
        assert_eq!(&a[..], b);
    }
}

///
/// Groups an iterator of dates by month.
///
trait ByMonth: DateIterator + Sized {
    fn by_month(self) -> GroupBy<u32, Self, fn(&NaiveDate) -> u32> {
        self.group_by(NaiveDate::month as fn(&NaiveDate) -> u32)
    }
}

impl<It> ByMonth for It where It: DateIterator {}

fn test_by_month() {
    let mut months = dates_in_year(2013).by_month();
    {
        for (month, (_, date)) in (1..13).zip(months.by_ref()) {
            assert_eq!(date[0], NaiveDate::from_ymd(2013, month, 1));
        }
    }
    assert_eq!(months.next(), None);
}

///
/// Groups an iterator of dates by week.
///

trait ByWeek: DateIterator + Sized {
    fn by_week(self) -> GroupBy<u32, Self, fn(&NaiveDate) -> u32> {
        self.group_by(to_iso_week)
    }
}

impl<It> ByWeek for It where It: DateIterator {}

fn to_iso_week(date: &NaiveDate) -> u32 {
    // We go forward one day because `isoweekdate` considers the week to start on a Monday.
    date.succ().isoweekdate().1
}

fn test_isoweekdate() {
    fn weeks_uniq(year: i32) -> Vec<((i32, u32), u32)> {
        let mut weeks = dates_in_year(year).map(|d| d.isoweekdate())
            .map(|(y,w,_)| (y,w));
        let mut result = vec![];
        let mut accum = (weeks.next().unwrap(), 1);
        for yw in weeks {
            if accum.0 == yw {
                accum.1 += 1;
            } else {
                result.push(accum);
                accum = (yw, 1);
            }
        }
        result.push(accum);
        result
    }

    let wu_1984 = weeks_uniq(1984);
    assert_eq!(&wu_1984[..2], &[((1983, 52), 1), ((1984, 1), 7)]);
    assert_eq!(&wu_1984[wu_1984.len()-2..], &[((1984, 52), 7), ((1985, 1), 1)]);

    let wu_2013 = weeks_uniq(2013);
    assert_eq!(&wu_2013[..2], &[((2013, 1), 6), ((2013, 2), 7)]);
    assert_eq!(&wu_2013[wu_2013.len()-2..], &[((2013, 52), 7), ((2014, 1), 2)]);

    let wu_2015 = weeks_uniq(2015);
    assert_eq!(&wu_2015[..2], &[((2015, 1), 4), ((2015, 2), 7)]);
    assert_eq!(&wu_2015[wu_2015.len()-2..], &[((2015, 52), 7), ((2015, 53), 4)]);
}

fn test_by_week() {
    let mut weeks = dates_in_year(2013).by_week();
    assert_eq!(
        &*weeks.next().unwrap().1,
        &[
            NaiveDate::from_ymd(2013, 1, 1),
            NaiveDate::from_ymd(2013, 1, 2),
            NaiveDate::from_ymd(2013, 1, 3),
            NaiveDate::from_ymd(2013, 1, 4),
            NaiveDate::from_ymd(2013, 1, 5),
        ]
    );
    assert_eq!(
        &*weeks.next().unwrap().1,
        &[
            NaiveDate::from_ymd(2013, 1, 6),
            NaiveDate::from_ymd(2013, 1, 7),
            NaiveDate::from_ymd(2013, 1, 8),
            NaiveDate::from_ymd(2013, 1, 9),
            NaiveDate::from_ymd(2013, 1, 10),
            NaiveDate::from_ymd(2013, 1, 11),
            NaiveDate::from_ymd(2013, 1, 12),
        ]
    );
    assert_eq!(weeks.next().unwrap().1[0], NaiveDate::from_ymd(2013, 1, 13));
}

/// The number of columns per day in the formatted output.
const COLS_PER_DAY: u32 = 3;

/// The number of columns per week in the formatted output.
const COLS_PER_WEEK: u32 = 7 * COLS_PER_DAY;

///
/// Formats an iterator of weeks into an iterator of strings.
///
trait FormatWeeks: Iterator<Item=Vec<NaiveDate>> + Sized {
    fn format_weeks(self) -> std::iter::Map<Self, fn(Vec<NaiveDate>) -> String> {
        self.map(format_week as fn(Vec<NaiveDate>) -> String)
    }
}

impl<It> FormatWeeks for It where It: Iterator<Item=Vec<NaiveDate>> {}

fn format_week(week: Vec<NaiveDate>) -> String {
    let mut buf = String::with_capacity((COLS_PER_DAY * COLS_PER_WEEK + 2) as usize);

    // Insert enough filler to align the first day with its respective day-of-week.
    let start_day = week[0].weekday().num_days_from_sunday();
    buf.extend(spaces((COLS_PER_DAY * start_day) as usize));

    // Format each day into its own cell and append to target string.
    // NOTE: Three things: first, `days` is lazy, unlike with the D version.  Second, we can't directly extend a `String` with an `Iterator<Item=String>`, hence we use `foreach` from `itertools`.  Third, because `into_iter` *consumes* the subject, we have to get the length of the vec *before* that.
    let num_days = week.len();
    let mut days = week.into_iter().map(|d| format!(" {:>2}", d.day()));
    days.foreach(|ds| buf.push_str(&ds));

    // Insert more filler at the end to fill up the remainder of the week, if its a short week (e.g. at the end of the month).
    buf.extend(spaces((COLS_PER_DAY * (7 - start_day - num_days as u32)) as usize));
    buf
}

fn test_format_weeks() {
    let jan_2013 = dates_in_year(2013)
        .by_month().next() // pick January 2013 for testing purposes
        // NOTE: This `map` is because `next` returns an `Option<_>`.
        .map(|(_, month)| month.into_iter()
            .by_week()
            .map(|(_, weeks)| weeks)
            .format_weeks()
            .collect::<Vec<_>>()
            .connect("\n"));

    assert_eq!(
        jan_2013.as_ref().map(|s| &**s),
        Some("        1  2  3  4  5\n\
           \x20 6  7  8  9 10 11 12\n\
           \x2013 14 15 16 17 18 19\n\
           \x2020 21 22 23 24 25 26\n\
           \x2027 28 29 30 31      ")
    );
}

///
/// Formats the name of a month, centered on COLS_PER_WEEK.
///
fn month_title(month: u32) -> String {
    const MONTH_NAMES: &'static [&'static str] = &[
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ];
    assert_eq!(MONTH_NAMES.len(), 12);

    // Determine how many spaces before and after the month name we need to center it over the formatted weeks in the month.
    let name = MONTH_NAMES[(month - 1) as usize];
    assert!(name.len() < COLS_PER_WEEK as usize);
    let before = (COLS_PER_WEEK as usize - name.len()) / 2;
    let after = COLS_PER_WEEK as usize - name.len() - before;

    // NOTE: Being slightly more verbose to avoid extra allocations.
    let mut result = String::with_capacity(COLS_PER_WEEK as usize);
    result.extend(spaces(before));
    result.push_str(name);
    result.extend(spaces(after));
    result
}

fn test_month_title() {
    assert_eq!(month_title(1).len(), COLS_PER_WEEK as usize);
}

///
/// Formats a month.
///

// NOTE: Here's a *perfect* example of just *why* people want Rust to get abstract return types.  Do note that if we did the `join("\n")` in this function, the return type would just be `String`.

trait FormatMonth: DateIterator + Sized {
    fn format_month(self)
    -> /* *deep breath* */
        std::iter::Chain<
            std::option::IntoIter<String>,
            std::iter::Map<
                std::iter::Map<
                    GroupBy<
                        u32,
                        std::iter::Peekable<Self>,
                        fn(&NaiveDate) -> u32
                    >,
                    fn((u32, Vec<NaiveDate>)) -> Vec<NaiveDate>
                >,
                fn(Vec<NaiveDate>) -> String
            >
        >
    {
        let mut month_days = self.peekable();
        let title = month_title(month_days.peek().unwrap().month());

        fn just_week((_, week): (u32, Vec<NaiveDate>)) -> Vec<NaiveDate> {
            week
        }

        Some(title).into_iter()
            .chain(month_days.by_week()
                .map(just_week as fn((u32, Vec<NaiveDate>)) -> Vec<NaiveDate>)
                .map(format_week as fn(Vec<NaiveDate>) -> String))
    }
}

impl<It> FormatMonth for It where It: DateIterator {}

fn test_format_month() {
    let month_fmt = dates_in_year(2013)
        .by_month().next() // Pick January as a test case
        .map(|(_, days)| days.into_iter()
            .format_month()
            .join("\n"));

    assert_eq!(
        month_fmt.as_ref().map(|s| &**s),
        Some("       January       \n\
           \x20       1  2  3  4  5\n\
           \x20 6  7  8  9 10 11 12\n\
           \x2013 14 15 16 17 18 19\n\
           \x2020 21 22 23 24 25 26\n\
           \x2027 28 29 30 31      ")
    );
}


///
/// Formats an iterator of months.
///

// NOTE: Yes, this really is all just to abstract a single method call.

trait FormatMonths: Iterator + Sized
where Self::Item: DateIterator {
    fn format_months(self)
    ->
        std::iter::Map<
            Self,
            fn(Self::Item)
            -> std::iter::Chain<
                std::option::IntoIter<String>,
                std::iter::Map<
                    std::iter::Map<
                        GroupBy<
                            u32,
                            std::iter::Peekable<Self::Item>,
                            fn(&NaiveDate) -> u32
                        >,
                        fn((u32, Vec<NaiveDate>)) -> Vec<NaiveDate>
                    >,
                    fn(Vec<NaiveDate>) -> String
                >
            >
        >
    {
        let f: fn(Self::Item)
            -> std::iter::Chain<
                std::option::IntoIter<String>,
                std::iter::Map<
                    std::iter::Map<
                        GroupBy<
                            u32,
                            std::iter::Peekable<Self::Item>,
                            fn(&NaiveDate) -> u32
                        >,
                        fn((u32, Vec<NaiveDate>)) -> Vec<NaiveDate>
                    >,
                    fn(Vec<NaiveDate>) -> String
                >
            > = Self::Item::format_month;
        self.map(f)
    }
}

impl<It> FormatMonths for It where It: Iterator, It::Item: DateIterator {}

///
/// Takes an iterator of iterators of strings; the sub-iterators are consumed
/// in lock-step, with their elements joined together.
///
trait PasteBlocks: Iterator + Sized
where Self::Item: Iterator<Item=String> {
    fn paste_blocks(self, sep_width: usize) -> PasteBlocksIter<Self::Item> {
        PasteBlocksIter {
            iters: self.collect(),
            cache: vec![],
            col_widths: None,
            sep_width: sep_width,
        }
    }
}

impl<It> PasteBlocks for It where It: Iterator, It::Item: Iterator<Item=String> {}

struct PasteBlocksIter<StrIt>
where StrIt: Iterator<Item=String> {
    iters: Vec<StrIt>,
    cache: Vec<Option<String>>,
    col_widths: Option<Vec<usize>>,
    sep_width: usize,
}

impl<StrIt> Iterator for PasteBlocksIter<StrIt>
where StrIt: Iterator<Item=String> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        self.cache.clear();

        // `cache` is now the next line from each iterator.
        self.cache.extend(self.iters.iter_mut().map(|it| it.next()));

        // If every line in `cache` is `None`, we have nothing further to do.
        if self.cache.iter().all(|e| e.is_none()) { return None }

        // Get the column widths if we haven't already.
        let col_widths = match self.col_widths {
            Some(ref v) => &**v,
            None => {
                self.col_widths = Some(self.cache.iter()
                    .map(|ms| ms.as_ref().map(|s| s.len()).unwrap_or(0))
                    .collect());
                &**self.col_widths.as_ref().unwrap()
            }
        };

        // Fill in any `None`s with spaces.
        let mut parts = col_widths.iter().cloned().zip(self.cache.iter_mut())
            .map(|(w,ms)| ms.take().unwrap_or_else(|| spaces(w).collect()));

        // Join them all together.
        let first = parts.next().unwrap_or(String::new());
        let sep_width = self.sep_width;
        Some(parts.fold(first, |mut accum, next| {
            accum.extend(spaces(sep_width));
            accum.push_str(&next);
            accum
        }))
    }
}

fn test_paste_blocks() {
    let row = dates_in_year(2013)
        .by_month().map(|(_, days)| days.into_iter())
        .take(3)
        .format_months()
        .paste_blocks(1)
        .join("\n");
    assert_eq!(
        &*row,
        "       January              February                March        \n\
      \x20       1  2  3  4  5                  1  2                  1  2\n\
      \x20 6  7  8  9 10 11 12   3  4  5  6  7  8  9   3  4  5  6  7  8  9\n\
      \x2013 14 15 16 17 18 19  10 11 12 13 14 15 16  10 11 12 13 14 15 16\n\
      \x2020 21 22 23 24 25 26  17 18 19 20 21 22 23  17 18 19 20 21 22 23\n\
      \x2027 28 29 30 31        24 25 26 27 28        24 25 26 27 28 29 30\n\
      \x20                                            31                  "
    );
}

///
/// Produces an iterator that yields `n` elements at a time.
///
trait Chunks: Iterator + Sized {
    fn chunks(self, n: usize) -> ChunksIter<Self> {
        assert!(n > 0);
        ChunksIter {
            it: self,
            n: n,
        }
    }
}

impl<It> Chunks for It where It: Iterator {}

struct ChunksIter<It>
where It: Iterator {
    it: It,
    n: usize,
}

// NOTE: `chunks` in Rust is more-or-less impossible without overhead of some kind.  Aliasing rules mean you need to add dynamic borrow checking, and the design of `Iterator` means that you need to have the iterator's state kept in an allocation that is jointly owned by the iterator itself and the sub-iterator.  As such, I've chosen to cop-out and just heap-allocate each chunk.

impl<It> Iterator for ChunksIter<It>
where It: Iterator {
    type Item = Vec<It::Item>;

    fn next(&mut self) -> Option<Vec<It::Item>> {
        let first = match self.it.next() {
            Some(e) => e,
            None => return None
        };

        let mut result = Vec::with_capacity(self.n);
        result.push(first);

        Some(self.it.by_ref().take(self.n-1)
            .fold(result, |mut acc, next| { acc.push(next); acc }))
    }
}

fn test_chunks() {
    let r = &[1, 2, 3, 4, 5, 6, 7];
    let c = r.iter().cloned().chunks(3).collect::<Vec<_>>();
    assert_eq!(&*c, &[vec![1, 2, 3], vec![4, 5, 6], vec![7]]);
}

///
/// Formats a year.
///
fn format_year(year: i32, months_per_row: usize) -> String {
    const COL_SPACING: usize = 1;

    // Start by generating all dates for the given year.
    dates_in_year(year)

        // Group them by month and throw away month number.
        .by_month().map(|(_, days)| days.into_iter())

        // Group the months into horizontal rows.
        .chunks(months_per_row)

        // Format each row
        .map(|r| r.into_iter()
            // By formatting each month
            .format_months()

            // Horizontally pasting each respective month's lines together.
            .paste_blocks(COL_SPACING)
            .join("\n")
        )

        // Insert a blank line between each row
        .join("\n\n")
}

fn test_format_year() {
    const MONTHS_PER_ROW: usize = 3;

    macro_rules! assert_eq_cal {
        ($lhs:expr, $rhs:expr) => {
            if $lhs != $rhs {
                println!("got:\n```\n{}\n```\n", $lhs.replace(" ", "."));
                println!("expected:\n```\n{}\n```", $rhs.replace(" ", "."));
                panic!("calendars didn't match!");
            }
        }
    }

    assert_eq_cal!(&format_year(1984, MONTHS_PER_ROW), "\
\x20      January              February                March        \n\
\x20 1  2  3  4  5  6  7            1  2  3  4               1  2  3\n\
\x20 8  9 10 11 12 13 14   5  6  7  8  9 10 11   4  5  6  7  8  9 10\n\
\x2015 16 17 18 19 20 21  12 13 14 15 16 17 18  11 12 13 14 15 16 17\n\
\x2022 23 24 25 26 27 28  19 20 21 22 23 24 25  18 19 20 21 22 23 24\n\
\x2029 30 31              26 27 28 29           25 26 27 28 29 30 31\n\
\n\
\x20       April                  May                  June         \n\
\x20 1  2  3  4  5  6  7         1  2  3  4  5                  1  2\n\
\x20 8  9 10 11 12 13 14   6  7  8  9 10 11 12   3  4  5  6  7  8  9\n\
\x2015 16 17 18 19 20 21  13 14 15 16 17 18 19  10 11 12 13 14 15 16\n\
\x2022 23 24 25 26 27 28  20 21 22 23 24 25 26  17 18 19 20 21 22 23\n\
\x2029 30                 27 28 29 30 31        24 25 26 27 28 29 30\n\
\n\
\x20       July                 August               September      \n\
\x20 1  2  3  4  5  6  7            1  2  3  4                     1\n\
\x20 8  9 10 11 12 13 14   5  6  7  8  9 10 11   2  3  4  5  6  7  8\n\
\x2015 16 17 18 19 20 21  12 13 14 15 16 17 18   9 10 11 12 13 14 15\n\
\x2022 23 24 25 26 27 28  19 20 21 22 23 24 25  16 17 18 19 20 21 22\n\
\x2029 30 31              26 27 28 29 30 31     23 24 25 26 27 28 29\n\
\x20                                            30                  \n\
\n\
\x20      October              November              December       \n\
\x20    1  2  3  4  5  6               1  2  3                     1\n\
\x20 7  8  9 10 11 12 13   4  5  6  7  8  9 10   2  3  4  5  6  7  8\n\
\x2014 15 16 17 18 19 20  11 12 13 14 15 16 17   9 10 11 12 13 14 15\n\
\x2021 22 23 24 25 26 27  18 19 20 21 22 23 24  16 17 18 19 20 21 22\n\
\x2028 29 30 31           25 26 27 28 29 30     23 24 25 26 27 28 29\n\
\x20                                            30 31               ");

    assert_eq_cal!(&format_year(2015, MONTHS_PER_ROW), "\
\x20      January              February                March        \n\
\x20             1  2  3   1  2  3  4  5  6  7   1  2  3  4  5  6  7\n\
\x20 4  5  6  7  8  9 10   8  9 10 11 12 13 14   8  9 10 11 12 13 14\n\
\x2011 12 13 14 15 16 17  15 16 17 18 19 20 21  15 16 17 18 19 20 21\n\
\x2018 19 20 21 22 23 24  22 23 24 25 26 27 28  22 23 24 25 26 27 28\n\
\x2025 26 27 28 29 30 31                        29 30 31            \n\
\n\
\x20       April                  May                  June         \n\
\x20          1  2  3  4                  1  2      1  2  3  4  5  6\n\
\x20 5  6  7  8  9 10 11   3  4  5  6  7  8  9   7  8  9 10 11 12 13\n\
\x2012 13 14 15 16 17 18  10 11 12 13 14 15 16  14 15 16 17 18 19 20\n\
\x2019 20 21 22 23 24 25  17 18 19 20 21 22 23  21 22 23 24 25 26 27\n\
\x2026 27 28 29 30        24 25 26 27 28 29 30  28 29 30            \n\
\x20                      31                                        \n\
\n\
\x20       July                 August               September      \n\
\x20          1  2  3  4                     1         1  2  3  4  5\n\
\x20 5  6  7  8  9 10 11   2  3  4  5  6  7  8   6  7  8  9 10 11 12\n\
\x2012 13 14 15 16 17 18   9 10 11 12 13 14 15  13 14 15 16 17 18 19\n\
\x2019 20 21 22 23 24 25  16 17 18 19 20 21 22  20 21 22 23 24 25 26\n\
\x2026 27 28 29 30 31     23 24 25 26 27 28 29  27 28 29 30         \n\
\x20                      30 31                                     \n\
\n\
\x20      October              November              December       \n\
\x20             1  2  3   1  2  3  4  5  6  7         1  2  3  4  5\n\
\x20 4  5  6  7  8  9 10   8  9 10 11 12 13 14   6  7  8  9 10 11 12\n\
\x2011 12 13 14 15 16 17  15 16 17 18 19 20 21  13 14 15 16 17 18 19\n\
\x2018 19 20 21 22 23 24  22 23 24 25 26 27 28  20 21 22 23 24 25 26\n\
\x2025 26 27 28 29 30 31  29 30                 27 28 29 30 31      ");
}

fn main() {
    // Run tests.
    test_spaces();
    test_dates_in_year();
    test_group_by();
    test_by_month();
    test_isoweekdate();
    test_by_week();
    test_format_weeks();
    test_month_title();
    test_format_month();
    test_paste_blocks();
    test_chunks();
    test_format_year();
}
