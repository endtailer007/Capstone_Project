# Books Database SQL Query Outputs


## High Rated In-Stock Books

**Description:** Demonstrates SELECT with WHERE condition filtering books having rating >= 4 and in_stock = 1.

```sql
SELECT 
    book_id,
    title,
    price_gbp,
    price_inr,
    rating,
    in_stock
FROM books
WHERE rating >= 4 AND in_stock = 1;
```

**Results (27 rows):**

| book_id | title                                                                    | price_gbp | price_inr | rating | in_stock |
| ------- | ------------------------------------------------------------------------ | --------- | --------- | ------ | -------- |
| 2       | Full Moon over Noah’s Ark: An Odyssey to Mount Ararat and Beyond         | 49.43     | 5214.86   | 4      | 1        |
| 8       | A Year in Provence (Provence #1)                                         | 56.88     | 6000.84   | 4      | 1        |
| 11      | 1,000 Places to See Before You Die                                       | 26.08     | 2751.44   | 5      | 1        |
| 12      | Sharp Objects                                                            | 47.82     | 5045.01   | 4      | 1        |
| 14      | The Past Never Ends                                                      | 56.5      | 5960.75   | 4      | 1        |
| 16      | The Murder of Roger Ackroyd (Hercule Poirot #4)                          | 44.1      | 4652.55   | 4      | 1        |
| 20      | A Time of Torment (Charlie Parker #14)                                   | 48.35     | 5100.92   | 5      | 1        |
| 23      | Murder at the 42nd Street Library (Raymond Ambler #1)                    | 54.36     | 5734.98   | 4      | 1        |
| 29      | What Happened on Beale Street (Secrets of the South Mysteries #2)        | 25.37     | 2676.54   | 5      | 1        |
| 30      | The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1) | 52.3      | 5517.65   | 5      | 1        |
| 31      | Delivering the Truth (Quaker Midwife Mystery #1)                         | 20.89     | 2203.9    | 4      | 1        |
| 32      | The Mysterious Affair at Styles (Hercule Poirot #1)                      | 24.8      | 2616.4    | 4      | 1        |
| 34      | The Silkworm (Cormoran Strike #2)                                        | 23.05     | 2431.78   | 5      | 1        |
| 39      | The No. 1 Ladies' Detective Agency (No. 1 Ladies' Detective Agency #1)   | 57.7      | 6087.35   | 4      | 1        |
| 40      | The Girl You Lost                                                        | 12.29     | 1296.59   | 5      | 1        |
| 46      | A Flight of Arrows (The Pathfinders #2)                                  | 55.53     | 5858.42   | 5      | 1        |
| 48      | Mrs. Houdini                                                             | 30.25     | 3191.38   | 5      | 1        |
| 49      | The Marriage of Opposites                                                | 28.08     | 2962.44   | 4      | 1        |
| 52      | A Paris Apartment                                                        | 39.01     | 4115.55   | 4      | 1        |
| 56      | World Without End (The Pillars of the Earth #2)                          | 32.97     | 3478.34   | 4      | 1        |
| 57      | The Passion of Dolssa                                                    | 28.32     | 2987.76   | 5      | 1        |
| 59      | Voyager (Outlander #3)                                                   | 21.07     | 2222.89   | 5      | 1        |
| 60      | The Red Tent                                                             | 35.66     | 3762.13   | 5      | 1        |
| 64      | Between Shades of Gray                                                   | 20.79     | 2193.34   | 5      | 1        |
| 65      | While You Were Mine                                                      | 41.32     | 4359.26   | 5      | 1        |
| 68      | Lost Among the Living                                                    | 27.7      | 2922.35   | 4      | 1        |
| 69      | A Spy's Devotion (The Regency Spies of London #1)                        | 16.97     | 1790.33   | 5      | 1        |

---

## Distinct Star Ratings in Database

**Description:** Demonstrates DISTINCT to find unique rating scores stored in the books table.

```sql
SELECT DISTINCT 
    rating
FROM books
ORDER BY rating ASC;
```

**Results (5 rows):**

| rating |
| ------ |
| 1      |
| 2      |
| 3      |
| 4      |
| 5      |

---

## Top 5 Most Expensive Books

**Description:** Demonstrates ORDER BY and LIMIT to find the 5 highest priced books.

```sql
SELECT 
    book_id,
    title,
    price_gbp,
    price_inr,
    rating
FROM books
ORDER BY price_inr DESC
LIMIT 5;
```

**Results (5 rows):**

| book_id | title                                                                  | price_gbp | price_inr | rating |
| ------- | ---------------------------------------------------------------------- | --------- | --------- | ------ |
| 26      | Boar Island (Anna Pigeon #19)                                          | 59.48     | 6275.14   | 3      |
| 39      | The No. 1 Ladies' Detective Agency (No. 1 Ladies' Detective Agency #1) | 57.7      | 6087.35   | 4      |
| 8       | A Year in Provence (Provence #1)                                       | 56.88     | 6000.84   | 4      |
| 14      | The Past Never Ends                                                    | 56.5      | 5960.75   | 4      |
| 61      | The Last Painting of Sara de Vos                                       | 55.55     | 5860.52   | 2      |

---

## Books with Category Names (INNER JOIN)

**Description:** Demonstrates INNER JOIN between books and categories tables, combined with WHERE and ORDER BY.

```sql
SELECT 
    b.book_id,
    b.title,
    c.category_name,
    b.price_gbp,
    b.price_inr,
    b.rating
FROM books b
INNER JOIN categories c ON b.category_id = c.category_id
WHERE c.category_name = 'Mystery'
ORDER BY b.price_inr ASC
LIMIT 5;
```

**Results (5 rows):**

| book_id | title                                  | category_name | price_gbp | price_inr | rating |
| ------- | -------------------------------------- | ------------- | --------- | --------- | ------ |
| 19      | Tastes Like Fear (DI Marnie Rome #3)   | Mystery       | 10.69     | 1127.79   | 1      |
| 25      | Hide Away (Eve Duncan #20)             | Mystery       | 11.84     | 1249.12   | 1      |
| 40      | The Girl You Lost                      | Mystery       | 12.29     | 1296.59   | 5      |
| 28      | Playing with Fire                      | Mystery       | 13.71     | 1446.41   | 3      |
| 18      | That Darkness (Gardiner and Renner #1) | Mystery       | 13.92     | 1468.56   | 1      |

---

## Category-Level Aggregations (JOIN + GROUP BY + ORDER BY)

**Description:** Demonstrates JOIN with GROUP BY to compute total books, average price in INR, and avg rating per category.

```sql
SELECT 
    c.category_id,
    c.category_name,
    COUNT(b.book_id) AS total_books,
    ROUND(AVG(b.price_inr), 2) AS avg_price_inr,
    ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp,
    ROUND(AVG(b.rating), 2) AS avg_rating,
    MIN(b.price_inr) AS min_price_inr,
    MAX(b.price_inr) AS max_price_inr
FROM categories c
INNER JOIN books b ON c.category_id = b.category_id
GROUP BY c.category_id, c.category_name
ORDER BY total_books DESC;
```

**Results (3 rows):**

| category_id | category_name      | total_books | avg_price_inr | avg_price_gbp | avg_rating | min_price_inr | max_price_inr |
| ----------- | ------------------ | ----------- | ------------- | ------------- | ---------- | ------------- | ------------- |
| 2           | Mystery            | 32          | 3346.36       | 31.72         | 2.94       | 1127.79       | 6275.14       |
| 3           | Historical Fiction | 26          | 3549.47       | 33.64         | 3.23       | 1753.41       | 5860.52       |
| 1           | Travel             | 11          | 4198.32       | 39.79         | 2.73       | 2448.66       | 6000.84       |

---
