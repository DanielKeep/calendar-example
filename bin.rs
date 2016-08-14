//! Derived from:
//! <https://raw.githubusercontent.com/quickfur/dcal/master/dcal.d>.
#![feature(conservative_impl_trait)]

extern crate chrono;
extern crate itertools;

use chrono::{Datelike, NaiveDate};
use itertools::Itertools;

///
/// Generates an iterator that yields exactly n spaces.
///
fn spaces(n: usize) -> impl Iterator<Item=char> {
    std::iter::repeat(' ').take(n)
}

#[cfg(test)]
#[test]
fn test_spaces() {
    assert_eq!(spaces(0).collect::<String>(), "");
    assert_eq!(spaces(10).collect::<String>(), "          ")
}

///
/// Returns an iterator built from the specified `next` function.
///
fn stepping<Step, Item>(step: Step) -> impl Iterator<Item=Item>
where Step: FnMut() -> Option<Item> {
    // This is what is known in D as a "Voldemort Type".
    struct It<F>(F);

    impl<F, I> Iterator for It<F> where F: FnMut() -> Option<I> {
        type Item = I;

        fn next(&mut self) -> Option<I> {
            (self.0)()
        }
    }

    It(step)
}

///
/// Returns an iterator of dates in a given year.
///
fn dates_in_year(year: i32) -> impl Iterator<Item=NaiveDate> {
    let mut cur = NaiveDate::from_ymd(year, 1, 1);
    stepping(move || {
        if cur.year() != year {
            None
        } else {
            let item = cur;
            cur = cur.succ();
            Some(item)
        }
    })
}

#[cfg(test)]
#[test]
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
/// This trait works around not being able to use `impl Trait` on trait methods
/// yet.
///
trait Apply: Sized {
    fn apply<F, R>(self, f: F) -> R where F: FnOnce(Self) -> R {
        f(self)
    }
}

impl<T> Apply for T {}

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
significant complication.  Part of the problem is the implicit sharing that has
to happen for it to work.

To give you an idea of the problems, see `Itertools::group_by_lazy`.
*/

#[cfg(test)]
#[test]
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
/// This trait is used to return lazy iterables.
///
/// The reason we need a trait is that the bound we're interested in is
/// *actually* implemented for *borrowed pointers* to the type we're returning,
/// and we can't express that with `impl Trait` syntax.
///
// trait LazyIterable<'a>: Sized {
//     type IntoIter: Iterator;
//     fn iter_lazy(&'a self) -> Self::IntoIter;
// }

// impl<'a, It> LazyIterable<'a> for It where It: 'a, &'a It: IntoIterator {
//     type IntoIter = <&'a It as IntoIterator>::IntoIter;
//     fn iter_lazy(&'a self) -> <&'a It as IntoIterator>::IntoIter {
//         self.into_iter()
//     }
// }

///
/// Groups an iterator of dates by month.
///
fn by_month<It: DateIterator>(it: It) -> impl Iterator<Item=(u32, Vec<NaiveDate>)> {
    it.group_by(NaiveDate::month)
}

fn by_month_lazy<It: DateIterator>(it: It)
-> itertools::GroupByLazy<u32, It, impl FnMut(&NaiveDate) -> u32> {
    it.group_by_lazy(NaiveDate::month)
}

#[cfg(test)]
#[test]
fn test_by_month() {
    let months = dates_in_year(2013).apply(by_month_lazy);
    let mut months = months.into_iter();
    {
        for (month, (_, mut group)) in (1..13).zip(months.by_ref()) {
            let first = group.next().unwrap();
            assert_eq!(first, NaiveDate::from_ymd(2013, month, 1));
        }
    }
    assert!(months.next().is_none());
}

///
/// Groups an iterator of dates by week.
///

fn by_week<It: DateIterator>(it: It) -> impl Iterator<Item=(u32, Vec<NaiveDate>)> {
    // We go forward one day because `isoweekdate` considers the week to start on a Monday.
    it.group_by(|date| date.succ().isoweekdate().1)
}

fn by_week_lazy<It: DateIterator>(it: It)
-> itertools::GroupByLazy<u32, It, impl FnMut(&NaiveDate) -> u32> {
    // We go forward one day because `isoweekdate` considers the week to start on a Monday.
    it.group_by_lazy(|date| date.succ().isoweekdate().1)
}

#[cfg(test)]
#[test]
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

#[cfg(test)]
#[test]
fn test_by_week() {
    let weeks = dates_in_year(2013).apply(by_week_lazy);
    let mut weeks = weeks.into_iter();
    assert_eq!(
        &*weeks.next().unwrap().1.collect_vec(),
        &[
            NaiveDate::from_ymd(2013, 1, 1),
            NaiveDate::from_ymd(2013, 1, 2),
            NaiveDate::from_ymd(2013, 1, 3),
            NaiveDate::from_ymd(2013, 1, 4),
            NaiveDate::from_ymd(2013, 1, 5),
        ]
    );
    assert_eq!(
        &*weeks.next().unwrap().1.collect_vec(),
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
    assert_eq!(weeks.next().unwrap().1.next().unwrap(), NaiveDate::from_ymd(2013, 1, 13));
}

/// The number of columns per day in the formatted output.
const COLS_PER_DAY: u32 = 3;

/// The number of columns per week in the formatted output.
const COLS_PER_WEEK: u32 = 7 * COLS_PER_DAY;

///
/// Formats an iterator of weeks into an iterator of strings.
///
fn format_weeks<It, IIt>(it: It) -> impl Iterator<Item=String>
where
    It: Iterator<Item=IIt>,
    IIt: IntoIterator<Item=NaiveDate>,
{
    it.map(format_week)
}

fn format_week<It: IntoIterator<Item=NaiveDate>>(week: It) -> String {
    {
        let mut buf = String::with_capacity((COLS_PER_DAY * COLS_PER_WEEK + 2) as usize);

        // Insert enough filler to align the first day with its respective day-
        // of-week.
        let mut week = week.into_iter().peekable();
        let start_day = week.peek().unwrap().weekday().num_days_from_sunday();
        buf.extend(spaces((COLS_PER_DAY * start_day) as usize));

        // Format each day into its own cell and append to target string.
        // **NOTE**: using `Write` here allows us to append to `buf` without any
        // intermediate allocations.
        let mut num_days = 0;
        for day in week {
            use std::fmt::Write;
            write!(buf, " {:>2}", day.day()).unwrap();
            num_days += 1;
        }

        // Insert more filler at the end to fill up the remainder of the week,
        // if its a short week (e.g. at the end of the month).
        buf.extend(spaces((COLS_PER_DAY * (7 - start_day - num_days)) as usize));
        buf
    }
}

#[cfg(test)]
#[test]
fn test_format_weeks() {
    let months_2013 = dates_in_year(2013)
        .apply(by_month_lazy);
    let jan_2013 = months_2013.into_iter()
        .next() // pick January 2013 for testing purposes
        // NOTE: This `map` is because `next` returns an `Option<_>`.
        .map(|(_, month)| {
            let weeks = month.apply(by_week_lazy);
            let weeks = weeks.into_iter();
            weeks
                .map(|(_, weeks)| {
                    let weeks: itertools::Group<_, _, _> = weeks;
                    weeks
                })

                // .apply(format_weeks) // <- borrowck problem
                .map(format_week) // <- no problem

                .join("\n")
        });

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

#[cfg(test)]
#[test]
fn test_month_title() {
    assert_eq!(month_title(1).len(), COLS_PER_WEEK as usize);
}

///
/// Formats a month.
///
fn format_month<It: DateIterator>(it: It) -> impl Iterator<Item=String> {
    let mut month_days = it.peekable();
    let title = month_title(month_days.peek().unwrap().month());

    Some(title).into_iter()
        .chain(month_days.apply(by_week)
            .map(|(_, week)| week)
            .apply(format_weeks))
}

#[cfg(test)]
#[test]
fn test_format_month() {
    let month_fmt = dates_in_year(2013)
        .apply(by_month).next() // Pick January as a test case
        .map(|(_, days)| days.into_iter()
            .apply(format_month)
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
fn format_months<It, DIt>(it: It) -> impl Iterator<Item=impl Iterator<Item=String>>
where
    It: Iterator<Item=DIt>,
    DIt: DateIterator,
{
    it.map(format_month)
}

///
/// Takes an iterator of iterators of strings; the sub-iterators are consumed
/// in lock-step, with their elements joined together.
///
/// Returns a callable that can be passed to `Apply::apply`.
///
fn paste_blocks<It, IIt>(sep_width: usize)
-> impl FnOnce(It) -> impl Iterator<Item=String>
where
    It: Iterator<Item=IIt>,
    IIt: Iterator<Item=String>,
{
    move |it| {
        let mut iters = it.collect_vec();
        let mut cache = vec![];
        let mut col_widths: Option<Vec<usize>> = None;

        stepping(move || {
            cache.clear();

            // `cache` is now the next line from each iterator.
            cache.extend(iters.iter_mut().map(|it| it.next()));

            // If every line in `cache` is `None`, we have nothing further to do.
            if cache.iter().all(|e| e.is_none()) { return None }

            // Get the column widths if we haven't already.
            let col_widths = match col_widths {
                Some(ref v) => &**v,
                None => {
                    col_widths = Some(cache.iter()
                        .map(|ms| ms.as_ref().map(|s| s.len()).unwrap_or(0))
                        .collect());
                    &**col_widths.as_ref().unwrap()
                }
            };

            // Fill in any `None`s with spaces.
            let mut parts = col_widths.iter().cloned().zip(cache.iter_mut())
                .map(|(w,ms)| ms.take().unwrap_or_else(|| spaces(w).collect()));

            // Join them all together.
            let first = parts.next().unwrap_or(String::new());
            let sep_width = sep_width;
            Some(parts.fold(first, |mut accum, next| {
                accum.extend(spaces(sep_width));
                accum.push_str(&next);
                accum
            }))
        })
    }
}

#[cfg(test)]
#[test]
fn test_paste_blocks() {
    let row = dates_in_year(2013)
        .apply(by_month).map(|(_, days)| days.into_iter())
        .take(3)
        .apply(format_months)
        .apply(paste_blocks(1))
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
fn chunks<It: Iterator>(n: usize) -> impl FnOnce(It) -> impl Iterator<Item=Vec<It::Item>> {
    /*
    **NOTE**: `chunks` in Rust is hard to implement without overhead of some
    kind.  Aliasing rules mean you need to add dynamic borrow checking, and the
    design of `Iterator` means that you need to have the iterator's state kept
    in an allocation that is jointly owned by the iterator itself and the
    sub-iterator.  As such, I've chosen to cop-out and just heap-allocate each
    chunk.
    */
    assert!(n > 0);
    move |mut it| {
        stepping(move || {
            let first = match it.next() {
                Some(e) => e,
                None => return None
            };

            let mut result = Vec::with_capacity(n);
            result.push(first);

            Some(it.by_ref().take(n-1)
                .fold(result, |mut acc, next| { acc.push(next); acc }))
        })
    }
}

#[cfg(test)]
#[test]
fn test_chunks() {
    let r = &[1, 2, 3, 4, 5, 6, 7];
    let c = r.iter().cloned().apply(chunks(3)).collect::<Vec<_>>();
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
        .apply(by_month).map(|(_, days)| days.into_iter())

        // Group the months into horizontal rows.
        .apply(chunks(months_per_row))

        // Format each row
        .map(|r| r.into_iter()
            // By formatting each month
            .apply(format_months)

            // Horizontally pasting each respective month's lines together.
            .apply(paste_blocks(COL_SPACING))
            .join("\n")
        )

        // Insert a blank line between each row
        .join("\n\n")
}

#[cfg(test)]
#[test]
fn test_format_year() {
    const MONTHS_PER_ROW: usize = 3;

    assert_eq!(&format_year(1984, MONTHS_PER_ROW), "\
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

    assert_eq!(&format_year(2015, MONTHS_PER_ROW), "\
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

#[cfg(not(test))]
fn main() {
    use std::env;

    let mut args = env::args();
    let exe = args.next().unwrap();

    let year = match args.next() {
        None => {
            println!("Usage: {} YEAR", exe);
            return
        }
        Some(arg) => arg.parse().ok().expect("expected integer year")
    };

    // Print the calendar.
    const MONTHS_PER_ROW: usize = 3;
    println!("{}", format_year(year, MONTHS_PER_ROW));
}
