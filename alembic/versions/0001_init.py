"""init schema"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("duration_sec", sa.Integer(), nullable=False),
        sa.Column("aspect_ratio", sa.String(length=16), nullable=False, server_default="16:9"),
        sa.Column("status", sa.Enum("SUBMITTED", "PLANNED", "SHOTS_QUEUED", "SHOTS_RENDERING", "SHOTS_DONE", "POSTPROCESSING", "DONE", "FAILED", name="jobstatus"), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("plan", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("result_path", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "shots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("jobs.id"), nullable=False),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column("duration_sec", sa.Integer(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("negative_prompt", sa.Text(), nullable=False),
        sa.Column("camera", sa.String(length=128), nullable=False),
        sa.Column("action", sa.String(length=128), nullable=False),
        sa.Column("environment", sa.String(length=128), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("resolution", sa.String(length=32), nullable=False),
        sa.Column("fps_internal", sa.Integer(), nullable=False),
        sa.Column("status", sa.Enum("QUEUED", "RENDERING", "DONE", "FAILED", name="shotstatus"), nullable=False),
        sa.Column("retries", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_retries", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("output_path", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("shots")
    op.drop_table("jobs")
    op.execute("DROP TYPE IF EXISTS shotstatus")
    op.execute("DROP TYPE IF EXISTS jobstatus")
