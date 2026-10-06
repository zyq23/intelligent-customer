"""北风电商数据 → Neo4j 图谱导入脚本

基于原项目 `往neo4j导数据.ipynb` 重构为可独立运行的脚本:
- 导入实体: Customer / Supplier / Category / Product / Order / Employee / Shipper / Review
- 导入关系: BELONGS_TO / SUPPLIED_BY / CONTAINS / PLACED / PROCESSED / SHIPPED_VIA / REPORTS_TO / ABOUT

用法:
    cd backend
    ./.venv/bin/python scripts/import_neo4j.py

配置:
    读取 .env 中的 NEO4J_URL / NEO4J_USERNAME / NEO4J_PASSWORD
    数据目录: app/graphrag/origin_data/exported_data
"""
import os
import sys
from pathlib import Path

# 项目根目录加入 PYTHONPATH
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

# 加载 .env 配置
load_dotenv(ROOT_DIR / ".env")

NEO4J_URI = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "csp_neo4j_2024")

DATA_DIR = ROOT_DIR / "app" / "graphrag" / "origin_data" / "exported_data"
BATCH_SIZE = 1000


class Neo4jImporter:
    """北风电商数据导入器"""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_pool_size=10,
            connection_timeout=30,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.driver.close()

    def setup_constraints(self):
        """创建唯一约束(幂等)"""
        constraints = {
            "Customer": "CustomerID",
            "Supplier": "SupplierID",
            "Category": "CategoryID",
            "Product": "ProductID",
            "Order": "OrderID",
            "Employee": "EmployeeID",
            "Shipper": "ShipperID",
            "Review": "ReviewID",
        }
        with self.driver.session() as session:
            for label, key in constraints.items():
                session.run(
                    f"CREATE CONSTRAINT {label.lower()}_id_unique IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
                )
            session.run(
                "CREATE INDEX product_name_index IF NOT EXISTS "
                "FOR (p:Product) ON (p.ProductName)"
            )
        print("[OK] 约束与索引已就绪")

    def _read(self, name: str) -> pd.DataFrame:
        path = DATA_DIR / f"{name}.csv"
        df = pd.read_csv(path, keep_default_na=False)
        df.columns = df.columns.str.strip()
        # 统一空值
        df = df.replace({"": None, "NULL": None, "null": None, "nan": None})
        return df

    def _clean_records(self, df: pd.DataFrame) -> list:
        return df.to_dict("records")

    def _batch(self, records: list):
        for i in range(0, len(records), BATCH_SIZE):
            yield records[i:i + BATCH_SIZE]

    # ---------- 实体导入 ----------
    def import_customers(self):
        df = self._read("customers")
        records = self._clean_records(df)
        query = """
        UNWIND $rows AS r
        MERGE (c:Customer {CustomerID: r.CustomerID})
        SET c.CompanyName = r.CompanyName, c.ContactName = r.ContactName,
            c.ContactTitle = r.ContactTitle, c.Address = r.Address,
            c.City = r.City, c.Region = r.Region, c.PostalCode = r.PostalCode,
            c.Country = r.Country, c.Phone = r.Phone, c.Fax = r.Fax
        """
        with self.driver.session() as session:
            for batch in self._batch(records):
                session.run(query, rows=batch)
        print(f"[OK] Customer: {len(records)}")

    def import_suppliers(self):
        df = self._read("suppliers")
        records = self._clean_records(df)
        query = """
        UNWIND $rows AS r
        MERGE (s:Supplier {SupplierID: r.SupplierID})
        SET s.CompanyName = r.CompanyName, s.ContactName = r.ContactName,
            s.ContactTitle = r.ContactTitle, s.Address = r.Address,
            s.City = r.City, s.Region = r.Region, s.PostalCode = r.PostalCode,
            s.Country = r.Country, s.Phone = r.Phone, s.Fax = r.Fax, s.HomePage = r.HomePage
        """
        with self.driver.session() as session:
            for batch in self._batch(records):
                session.run(query, rows=batch)
        print(f"[OK] Supplier: {len(records)}")

    def import_categories(self):
        df = self._read("categories")
        records = self._clean_records(df)
        query = """
        UNWIND $rows AS r
        MERGE (c:Category {CategoryID: r.CategoryID})
        SET c.CategoryName = r.CategoryName, c.Description = r.Description
        """
        with self.driver.session() as session:
            for batch in self._batch(records):
                session.run(query, rows=batch)
        print(f"[OK] Category: {len(records)}")

    def import_products(self):
        df = self._read("products")
        records = self._clean_records(df)
        query = """
        UNWIND $rows AS r
        MERGE (p:Product {ProductID: r.ProductID})
        SET p.ProductName = r.ProductName, p.SupplierID = r.SupplierID,
            p.CategoryID = r.CategoryID, p.QuantityPerUnit = r.QuantityPerUnit,
            p.UnitPrice = toFloat(r.UnitPrice), p.UnitsInStock = toInteger(r.UnitsInStock),
            p.UnitsOnOrder = toInteger(r.UnitsOnOrder), p.ReorderLevel = toInteger(r.ReorderLevel),
            p.Discontinued = toBoolean(r.Discontinued), p.CategoryName = r.CategoryName,
            p.SupplierName = r.SupplierName
        """
        with self.driver.session() as session:
            for batch in self._batch(records):
                session.run(query, rows=batch)
        print(f"[OK] Product: {len(records)}")

    def import_orders(self):
        df = self._read("orders")
        records = self._clean_records(df)
        query = """
        UNWIND $rows AS r
        MERGE (o:Order {OrderID: r.OrderID})
        SET o.CustomerID = r.CustomerID, o.EmployeeID = r.EmployeeID,
            o.OrderDate = r.OrderDate, o.RequiredDate = r.RequiredDate,
            o.ShippedDate = r.ShippedDate, o.ShipVia = r.ShipVia,
            o.Freight = toFloat(r.Freight), o.ShipName = r.ShipName,
            o.ShipAddress = r.ShipAddress, o.ShipCity = r.ShipCity,
            o.ShipRegion = r.ShipRegion, o.ShipPostalCode = r.ShipPostalCode,
            o.ShipCountry = r.ShipCountry, o.CustomerName = r.CustomerName,
            o.ShipperName = r.ShipperName
        """
        with self.driver.session() as session:
            for batch in self._batch(records):
                session.run(query, rows=batch)
        print(f"[OK] Order: {len(records)}")

    def import_employees(self):
        df = self._read("employees")
        records = self._clean_records(df)
        query = """
        UNWIND $rows AS r
        MERGE (e:Employee {EmployeeID: r.EmployeeID})
        SET e.LastName = r.LastName, e.FirstName = r.FirstName, e.Title = r.Title,
            e.TitleOfCourtesy = r.TitleOfCourtesy, e.BirthDate = r.BirthDate,
            e.HireDate = r.HireDate, e.Address = r.Address, e.City = r.City,
            e.Region = r.Region, e.PostalCode = r.PostalCode, e.Country = r.Country,
            e.HomePhone = r.HomePhone, e.Extension = r.Extension, e.Notes = r.Notes,
            e.ReportsTo = r.ReportsTo
        """
        with self.driver.session() as session:
            for batch in self._batch(records):
                session.run(query, rows=batch)
        print(f"[OK] Employee: {len(records)}")

    def import_shippers(self):
        df = self._read("shippers")
        records = self._clean_records(df)
        query = """
        UNWIND $rows AS r
        MERGE (s:Shipper {ShipperID: r.ShipperID})
        SET s.CompanyName = r.CompanyName, s.Phone = r.Phone
        """
        with self.driver.session() as session:
            for batch in self._batch(records):
                session.run(query, rows=batch)
        print(f"[OK] Shipper: {len(records)}")

    def import_reviews(self):
        df = self._read("reviews")
        records = self._clean_records(df)
        query = """
        UNWIND $rows AS r
        MERGE (rv:Review {ReviewID: r.ReviewID})
        SET rv.ProductID = r.ProductID, rv.ProductName = r.ProductName,
            rv.CustomerID = r.CustomerID, rv.CustomerName = r.CustomerName,
            rv.Rating = toFloat(r.Rating), rv.ReviewText = r.ReviewText,
            rv.ReviewDate = r.ReviewDate, rv.CategoryName = r.CategoryName
        """
        with self.driver.session() as session:
            for batch in self._batch(records):
                session.run(query, rows=batch)
        print(f"[OK] Review: {len(records)}")

    # ---------- 关系导入 ----------
    def import_relationships(self):
        with self.driver.session() as session:
            # 1. Product - BELONGS_TO -> Category
            session.run("""
            MATCH (p:Product), (c:Category)
            WHERE toInteger(p.CategoryID) = toInteger(c.CategoryID)
            MERGE (p)-[:BELONGS_TO]->(c)
            """)

            # 2. Product - SUPPLIED_BY -> Supplier
            session.run("""
            MATCH (p:Product), (s:Supplier)
            WHERE toInteger(p.SupplierID) = toInteger(s.SupplierID)
            MERGE (p)-[:SUPPLIED_BY]->(s)
            """)

            # 3. Customer - PLACED -> Order
            session.run("""
            MATCH (c:Customer), (o:Order)
            WHERE c.CustomerID = o.CustomerID
            MERGE (c)-[:PLACED]->(o)
            """)

            # 4. Employee - PROCESSED -> Order
            session.run("""
            MATCH (e:Employee), (o:Order)
            WHERE toInteger(e.EmployeeID) = toInteger(o.EmployeeID)
            MERGE (e)-[:PROCESSED]->(o)
            """)

            # 5. Order - SHIPPED_VIA -> Shipper
            session.run("""
            MATCH (o:Order), (s:Shipper)
            WHERE toInteger(o.ShipVia) = toInteger(s.ShipperID)
            MERGE (o)-[:SHIPPED_VIA]->(s)
            """)

            # 6. Employee - REPORTS_TO -> Employee
            session.run("""
            MATCH (e1:Employee), (e2:Employee)
            WHERE toInteger(e1.ReportsTo) = toInteger(e2.EmployeeID)
            MERGE (e1)-[:REPORTS_TO]->(e2)
            """)

            # 7. Review - ABOUT -> Product
            session.run("""
            MATCH (rv:Review), (p:Product)
            WHERE toInteger(rv.ProductID) = toInteger(p.ProductID)
            MERGE (rv)-[:ABOUT]->(p)
            """)

        # 8. Order - CONTAINS -> Product (带数量/单价/折扣属性)
        df = self._read("order_details")
        records = self._clean_records(df)
        query = """
        UNWIND $rows AS r
        MATCH (o:Order {OrderID: r.OrderID})
        MATCH (p:Product {ProductID: r.ProductID})
        MERGE (o)-[rel:CONTAINS]->(p)
        SET rel.Quantity = toInteger(r.Quantity),
            rel.UnitPrice = toFloat(r.UnitPrice),
            rel.Discount = toFloat(r.Discount)
        """
        with self.driver.session() as session:
            for batch in self._batch(records):
                session.run(query, rows=batch)
        print(f"[OK] CONTAINS 关系: {len(records)}")

    def import_all(self):
        self.setup_constraints()
        self.import_customers()
        self.import_suppliers()
        self.import_categories()
        self.import_products()
        self.import_orders()
        self.import_employees()
        self.import_shippers()
        self.import_reviews()
        self.import_relationships()


def main():
    if not NEO4J_PASSWORD or NEO4J_PASSWORD == "password":
        print("⚠️ 请先在 .env 中配置 NEO4J_PASSWORD")
        return 1
    print(f"连接: {NEO4J_URI} 用户: {NEO4J_USER}")
    print(f"数据目录: {DATA_DIR}")
    try:
        with Neo4jImporter() as importer:
            importer.import_all()
        print("\n🎉 全部数据导入完成!")
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
