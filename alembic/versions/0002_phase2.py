"""phase2 features"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_phase2"
down_revision = "0001_init"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("character_profile", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("jobs", sa.Column("postprocess_config", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("shots", sa.Column("continuity_mode", sa.String(length=32), nullable=True))
    op.add_column("shots", sa.Column("input_ref_image_path", sa.Text(), nullable=True))
    op.alter_column("shots", "max_retries", existing_type=sa.Integer(), server_default="3")


def downgrade() -> None:
    op.drop_column("shots", "input_ref_image_path")
    op.drop_column("shots", "continuity_mode")
    op.drop_column("jobs", "postprocess_config")
    op.drop_column("jobs", "character_profile")
